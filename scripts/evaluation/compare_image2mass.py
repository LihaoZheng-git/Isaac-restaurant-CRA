"""Compare two mass regressors offline without changing robot runtime policy.

Use model-native preprocessing on the same images and SI dimensions. ABO
dimensions/mass are catalog labels; Isaac fixture geometry, object masks, and
mass are oracle labels. Neither domain establishes sensor-derived robot mass
accuracy. No class name or reference mass is passed to either neural network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.visual_mass_model import (
    load_visual_mass_checkpoint,
    visual_mass_transform,
)
from scripts.evaluation.image2mass_compat import (
    Image2MassOffline,
    KG_PER_POUND,
    METRES_PER_INCH,
)


def summarize_rows(rows: list[dict], key: str) -> dict:
    """Magnitude and ordering metrics on the exact same valid pairs."""
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return {"count": 0}
    reference = np.asarray([r["reference_mass_kg"] for r in valid])
    predicted = np.asarray([r[key] for r in valid])
    errors = np.abs(predicted - reference)
    pairs = correct = 0
    for i in range(len(valid)):
        for j in range(i):
            # Omit repeat views of an object and nearly equal labels. The
            # >=2x ratio is defined before looking at either prediction.
            if valid[i]["object_id"] == valid[j]["object_id"]:
                continue
            if max(reference[i], reference[j]) < 2 * min(reference[i], reference[j]):
                continue
            pairs += 1
            correct += int(
                (reference[i] - reference[j]) * (predicted[i] - predicted[j]) > 0
            )
    return {
        "count": len(valid),
        "mae_kg": float(errors.mean()),
        "median_absolute_error_kg": float(np.median(errors)),
        "mean_absolute_percentage_error": float((errors / reference).mean()),
        "spearman": float(spearmanr(reference, predicted).statistic),
        "twofold_pairs": pairs,
        "twofold_pair_order_accuracy": correct / pairs if pairs else None,
    }


def view_variation(rows: list[dict], key: str) -> dict:
    grouped = {}
    for row in rows:
        if row.get(key) is not None:
            grouped.setdefault(row["object_id"], []).append(row[key])
    ratios = [max(v) / min(v) for v in grouped.values() if len(v) >= 2]
    return {
        "objects": len(ratios),
        "median_max_min_ratio": float(np.median(ratios)) if ratios else None,
        "worst_max_min_ratio": max(ratios) if ratios else None,
    }


def independent_objects(rows: list[dict], key: str) -> list[dict]:
    """Collapse repeated views before the primary ordering comparison."""
    grouped = {}
    for row in rows:
        if row.get(key) is not None:
            grouped.setdefault(row["object_id"], []).append(row)
    result = []
    for object_id, views in grouped.items():
        references = {view["reference_mass_kg"] for view in views}
        if len(references) != 1:
            raise ValueError("inconsistent reference mass for repeated object")
        result.append(
            {
                "object_id": object_id,
                "reference_mass_kg": references.pop(),
                key: float(np.median([view[key] for view in views])),
            }
        )
    return result


def crop_object(rgb: np.ndarray, mask: np.ndarray, fill: int) -> np.ndarray:
    """Same selected object bounds, model-native outside-mask background."""
    if mask.shape != rgb.shape[:2] or not mask.any():
        raise ValueError("valid matching object mask is required")
    yy, xx = np.where(mask)
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    return np.where(mask[y0:y1, x0:x1, None], rgb[y0:y1, x0:x1], fill).astype(np.uint8)


def compare(args: argparse.Namespace) -> dict:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    candidate = Image2MassOffline(args.exports, device)
    current, _ = load_visual_mass_checkpoint(args.checkpoint, map_location=device)
    transform = visual_mass_transform(training=False)

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def infer_current(rgb: np.ndarray, dims: np.ndarray) -> float:
        pixels = transform(Image.fromarray(rgb))[None].to(device)
        geometry = torch.tensor(dims, dtype=torch.float32, device=device)[None]
        with torch.inference_mode():
            return float(current.predict_mass_kg(pixels, geometry).item())

    parity = []
    for record in json.loads(args.parity_manifest.read_text("utf-8")):
        path = args.upstream / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("parity image hash mismatch")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        predicted = candidate.predict(
            rgb, np.asarray(record["dims_inches"]) * METRES_PER_INCH
        )
        pounds = predicted["mass_kg"] / KG_PER_POUND
        parity.append(
            {
                **record,
                "predicted_lb": pounds,
                "relative_difference": abs(pounds - record["lb"]) / record["lb"],
            }
        )
    (output / "parity.json").write_text(json.dumps(parity, indent=2), "utf-8")
    # Rounded published predictions are a numerical conformance check, not
    # proof of bitwise equivalence for every possible input.
    if (
        not parity
        or max(r["relative_difference"] for r in parity) > args.parity_tolerance
    ):
        raise RuntimeError(
            "published-prediction conformance gate failed; see parity.json"
        )

    rows = []
    collections = [
        ("abo", args.abo_manifest, "catalog_metadata", "catalog_metadata"),
        (
            "isaac_fixture",
            args.fixture_manifest,
            "oracle_authored_geometry",
            "oracle_authored_mass",
        ),
    ]
    # Warm both networks; report excludes startup/loading and all data I/O.
    warm = np.full((299, 299, 3), 255, np.uint8)
    warm[70:220, 70:220] = [40, 100, 200]
    infer_current(warm, np.asarray([0.1, 0.1, 0.1]))
    candidate.predict(warm, np.asarray([0.1, 0.1, 0.1]))
    for domain, manifest, geometry_basis, reference_basis in collections:
        items = [
            json.loads(line)
            for line in manifest.read_text("utf-8").splitlines()
            if line.strip()
        ]
        items = sorted(
            (item for item in items if item["split"] == "test"),
            key=lambda item: (item["object_id"], item.get("sample_id", "")),
        )
        for item in items:
            dims = np.asarray([item["width_m"], item["depth_m"], item["height_m"]])
            if domain == "abo":
                with Image.open(
                    manifest.parent / "images" / item["image_path"]
                ) as image:
                    rgb = np.asarray(image.convert("RGB"))
                current_rgb = candidate_rgb = rgb
                mask_basis = "original_catalog_image_no_new_segmentation"
            else:
                rgb = np.load(manifest.parent / item["rgb_path"], allow_pickle=False)
                mask = np.load(
                    manifest.parent / item["mask_path"], allow_pickle=False
                ).astype(bool)
                current_rgb = crop_object(rgb, mask, 127)
                candidate_rgb = crop_object(rgb, mask, 255)
                mask_basis = "oracle_rendered_instance_mask"
            synchronize()
            start = time.perf_counter()
            existing_mass = infer_current(current_rgb, dims)
            synchronize()
            current_ms = (time.perf_counter() - start) * 1000
            synchronize()
            start = time.perf_counter()
            error = None
            try:
                result = candidate.predict(candidate_rgb, dims)
            except ValueError as exc:
                result = {"mass_kg": None, "augmented_mass_kg": None}
                error = str(exc)
            synchronize()
            candidate_ms = (time.perf_counter() - start) * 1000
            row = {
                "domain": domain,
                "object_id": item["object_id"],
                "sample_id": item.get("sample_id", item["object_id"]),
                "class_for_report_only": item["semantic_class"],
                "dimensions_m": dims.tolist(),
                "geometry_provenance": geometry_basis,
                "mask_provenance": mask_basis,
                "reference_provenance": reference_basis,
                "reference_mass_kg": item["mass_kg"],
                "current_mass_kg": existing_mass,
                "image2mass_mass_kg": result["mass_kg"],
                "image2mass_augmented_mass_kg": result["augmented_mass_kg"],
                "image2mass_rejection": error,
                "current_ms": current_ms,
                "image2mass_ms": candidate_ms,
            }
            rows.append(row)
        print(f"completed {domain}: {len(items)} samples", flush=True)
    (output / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", "utf-8"
    )
    report = {
        "scope": "offline raw-regressor comparison; robot runtime unchanged; no wheels",
        "device": device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "input_hashes": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (args.abo_manifest, args.fixture_manifest, args.parity_manifest)
        ],
        "numeric_export_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.exports.iterdir())
            if path.is_file()
        },
        "parity_count": len(parity),
        "parity_tolerance": args.parity_tolerance,
        "parity_max_relative_difference": max(r["relative_difference"] for r in parity),
        "training_overlap": "ABO held out for current model; Image2Mass Amazon overlap unknown",
        "model_inputs": "RGB and SI width/depth/height only; no category or reference mass",
        "image2mass_units": "metres converted to inches; median pounds converted to kg",
        "domains": {},
    }
    for domain, _, _, _ in collections:
        selected = [row for row in rows if row["domain"] == domain]
        paired = [row for row in selected if row["image2mass_mass_kg"] is not None]
        result = {
            "samples": len(selected),
            "paired_samples": len(paired),
            "independent_objects": len({row["object_id"] for row in selected}),
            "image2mass_rejections": len(selected) - len(paired),
        }
        for model, key in (
            ("current", "current_mass_kg"),
            ("image2mass", "image2mass_mass_kg"),
        ):
            result[model] = summarize_rows(paired, key)
            result[model]["independent_object_metrics"] = summarize_rows(
                independent_objects(paired, key), key
            )
            result[model]["view_variation"] = view_variation(paired, key)
            latencies = [row[model + "_ms"] for row in paired]
            result[model]["latency_median_ms"] = float(np.median(latencies))
            result[model]["latency_p95_ms"] = float(np.percentile(latencies, 95))
        report["domains"][domain] = result
    (output / "report.json").write_text(json.dumps(report, indent=2), "utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--parity-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--abo-manifest", required=True, type=Path)
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parity-tolerance", default=0.0001, type=float)
    args = parser.parse_args()
    if not 0 < args.parity_tolerance <= 0.0001:
        parser.error("parity tolerance must be positive and at most 0.01%")
    print(json.dumps(compare(args), indent=2))
