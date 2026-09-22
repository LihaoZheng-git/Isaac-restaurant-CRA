"""Train and evaluate an offline MobileNetV3 visual-mass baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.abo_visual_mass import read_abo_subset_manifest
from robot.perception.visual_mass_model import (
    IMAGENET_WEIGHTS_NAME,
    IMAGENET_WEIGHTS_URL,
    VISUAL_MASS_ARCHITECTURE,
    VISUAL_MASS_MODEL_SCHEMA_VERSION,
    AboVisualMassDataset,
    VisualMassRegressor,
    fit_log_dimension_baseline,
    log_dimension_statistics,
    mass_regression_metrics,
    predict_log_dimension_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fine-tune-backbone", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, device):
    model.eval()
    targets = []
    predictions = []
    classes = []
    with torch.no_grad():
        for images, dimensions, log_mass, _, semantic_class in loader:
            prediction = model(images.to(device), dimensions.to(device))
            targets.append(log_mass)
            predictions.append(prediction.cpu())
            classes.extend(semantic_class)
    target = torch.cat(targets)
    predicted = torch.cat(predictions)
    overall = mass_regression_metrics(target, predicted)
    per_class = {}
    for name in sorted(set(classes)):
        indices = torch.tensor([value == name for value in classes])
        per_class[name] = asdict(mass_regression_metrics(target[indices], predicted[indices]))
    return overall, per_class


def _weight_file_metadata() -> dict[str, object]:
    filename = Path(urlparse(IMAGENET_WEIGHTS_URL).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        return {"cache_path": str(path), "sha256": None, "size_bytes": None}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"cache_path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    candidates = read_abo_subset_manifest(args.manifest)
    by_split = {
        split: tuple(item for item in candidates if item.split == split)
        for split in ("train", "validation", "test")
    }
    if any(not items for items in by_split.values()):
        raise ValueError("train, validation, and test splits must all be nonempty")
    mean, std = log_dimension_statistics(by_split["train"])
    datasets = {
        split: AboVisualMassDataset(
            items, args.image_root, training=split == "train"
        )
        for split, items in by_split.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    pretrained = not args.no_pretrained
    model = VisualMassRegressor(
        mean,
        std,
        pretrained=pretrained,
        freeze_backbone=not args.fine_tune_backbone,
    ).to(device)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_mae = math.inf
    best_epoch = 0
    checkpoint_path = args.output_dir / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, dimensions, log_mass, _, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images.to(device), dimensions.to(device))
            loss = loss_function(prediction, log_mass.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _ = _evaluate(model, loaders["validation"], device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "validation": asdict(validation),
            }
        )
        print(
            f"epoch {epoch:02d}: loss={history[-1]['train_loss']:.4f}; "
            f"val_mae={validation.mean_absolute_error_kg:.4f} kg"
        )
        if validation.mean_absolute_error_kg < best_validation_mae:
            best_validation_mae = validation.mean_absolute_error_kg
            best_epoch = epoch
            torch.save(
                {
                    "schema_version": VISUAL_MASS_MODEL_SCHEMA_VERSION,
                    "architecture": VISUAL_MASS_ARCHITECTURE,
                    "state_dict": model.state_dict(),
                    "dimension_mean": mean,
                    "dimension_std": std,
                    "pretrained": pretrained,
                    "backbone_frozen": not args.fine_tune_backbone,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation, validation_by_class = _evaluate(model, loaders["validation"], device)
    test, test_by_class = _evaluate(model, loaders["test"], device)
    test_target = torch.tensor(
        [math.log(item.mass_kg) for item in by_split["test"]]
    )
    train_target = torch.tensor(
        [math.log(item.mass_kg) for item in by_split["train"]]
    )
    median_prediction = torch.full_like(test_target, train_target.median())
    dimension_coefficients = fit_log_dimension_baseline(by_split["train"])
    dimension_prediction = predict_log_dimension_baseline(
        by_split["test"], dimension_coefficients
    )
    report = {
        "schema_version": VISUAL_MASS_MODEL_SCHEMA_VERSION,
        "architecture": VISUAL_MASS_ARCHITECTURE,
        "device": str(device),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "split_counts": {name: len(items) for name, items in by_split.items()},
        "dimension_mean": mean,
        "dimension_std": std,
        "pretrained_weights": {
            "used": pretrained,
            "name": IMAGENET_WEIGHTS_NAME if pretrained else None,
            "url": IMAGENET_WEIGHTS_URL if pretrained else None,
            "torchvision_code_license": "BSD-3-Clause",
            "weights_license_metadata": "not_provided",
            "training_dataset": "ImageNet-1K",
            **(_weight_file_metadata() if pretrained else {}),
        },
        "training_labels": {
            "dataset": "Amazon Berkeley Objects subset",
            "license": "CC BY 4.0",
            "provenance": "external_catalog_metadata",
            "runtime_evidence_status": "unavailable",
        },
        "validation": asdict(validation),
        "validation_by_class": validation_by_class,
        "test": asdict(test),
        "test_by_class": test_by_class,
        "comparison_baselines": {
            "global_train_median": asdict(
                mass_regression_metrics(test_target, median_prediction)
            ),
            "log_dimensions_least_squares": {
                **asdict(mass_regression_metrics(test_target, dimension_prediction)),
                "coefficients": dimension_coefficients.tolist(),
            },
        },
        "history": history,
        "limitations": [
            "ABO weights and dimensions are external catalog metadata claims.",
            "Some listings may describe a packaged set rather than one visible object.",
            "Catalog-image performance does not establish Isaac or robot runtime validity.",
            "Rare classes have too few independent products for stable per-class claims.",
            "The ImageNet pretrained-weight enum provides no explicit license field.",
        ],
    }
    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"best epoch: {best_epoch}")
    print(f"test MAE: {test.mean_absolute_error_kg:.4f} kg")
    print(f"test MAPE: {test.mean_absolute_percentage_error:.4f}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
