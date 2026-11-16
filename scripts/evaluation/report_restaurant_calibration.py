"""Archived-response study; neither generates controls nor changes native records."""

from dataclasses import asdict
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/restaurant-risk-calibration"
sys.path.insert(0, str(ROOT))
from robot.evaluation.restaurant_calibration_runtime import object_is_eligible
from robot.risk.restaurant_calibration import (
    RestaurantCalibrationConfig,
    RestaurantFactorWeights,
    evaluate_restaurant_calibration,
)

CONFIG = RestaurantCalibrationConfig(
    enabled=True,
    potential_energy_scale_j=100,
    kinetic_energy_scale_j=50,
    contact_power=2,
    context_floor=0.75,
    uncertainty_gain=0.25,
    weights=RestaurantFactorWeights(kinetic=1.5),
    human_protection_floor=0.8,
)


def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(
        n=len(x),
        minimum=float(x.min()),
        q10=float(np.quantile(x, 0.1)),
        median=float(np.median(x)),
        q90=float(np.quantile(x, 0.9)),
        maximum=float(x.max()),
        at_least_095=int((x >= 0.95).sum()),
        saturated=int((x >= 1 - 1e-9).sum()),
    )


def assess(row, config=CONFIG, human=False):
    obj = row["object"]
    d = obj["diagnostics"]
    # Original 0047241 static_prop_runtime uses its separate 120 s persistence
    # config. The top-level 3 s setting belongs to generic/human memory.
    maximum_age = (
        120.0 if d.get("static_prop_policy") else row["maximum_property_age_seconds"]
    )
    return evaluate_restaurant_calibration(
        obj["cra_control"]["cra"],
        d["factor_scores"],
        config,
        eligible=object_is_eligible(d)
        or (human and d.get("policy_return_count", 0) > 0),
        verified_human=human,
        source_energy_scales_j={"potential": 100.0, "kinetic": 100.0},
        contact_proxy=d.get("contact_proxy"),
        contact_interaction=obj["attributes"]["motion"]["route_crossing_likelihood"],
        timestamp_seconds=row["assessment_time"],
        maximum_contact_age_seconds=maximum_age,
    )


def extract_native_rows(paired_manifest, mission_manifest):
    """Fixture labels select evaluation rows only; they never enter scoring."""
    pair = json.loads(Path(paired_manifest).read_text())
    mission = json.loads(Path(mission_manifest).read_text())
    data, hashes = {}, {}

    def records(path):
        hashes[str(path)] = digest(path)
        with path.open("rb") as stream:
            if path.suffix == ".jsonl":
                yield from (json.loads(line) for line in stream if line.strip())
            else:
                yield from json.load(stream)

    def saved(record, obj):
        return dict(
            source=record["sensor_timestamp"],
            frame=record["sensor_frame_id"],
            assessment_time=obj["attributes"]["timestamp_seconds"],
            maximum_property_age_seconds=record["diagnostics"][
                "maximum_property_age_seconds"
            ],
            object=obj,
        )

    for label in ("bin", "crate"):
        folder = Path(pair[label]["folder"])
        parked = {(r["frame"], r["track"]) for r in pair[label]["parked"]["rows"]}
        ids = {key for _, key in parked}
        for phase, filename in (
            ("parked", "stationary-fields.json"),
            ("live", "live-fields.jsonl"),
        ):
            selected = []
            for r in records(folder / filename):
                for obj in r["objects"]:
                    qualifies = (
                        (r["sensor_frame_id"], obj["track_id"]) in parked
                        if phase == "parked"
                        else obj["track_id"] in ids
                        and obj["diagnostics"].get("static_prop_policy")
                    )
                    if qualifies:
                        selected.append(saved(r, obj))
            if len(selected) != pair[label][phase]["qualified"]:
                raise ValueError(
                    "short calibration selection differs from original audit"
                )
            data[f"calibration_{label}_{phase}"] = selected
    folder = Path(mission["source_directory"])
    coverage_path = folder / "contact-coverage.json"
    coverage = json.loads(coverage_path.read_text())
    hashes[str(coverage_path)] = digest(coverage_path)
    fresh = {(r["time"], r["id"]): r["prop"] for r in coverage["fresh_rows"]}
    names = {"ServiceContainer": "bin", "BrokenCrate": "crate"}
    for label in ("bin", "crate"):
        for phase in ("fresh", "retained"):
            data[f"archived_v3_{label}_{phase}"] = []
    data["archived_v3_verified_human"] = []
    source = folder / "live-fields.jsonl"
    for record in records(source):
        humans = {
            q["track_id"] for q in record["diagnostics"]["human_body_envelope_priors"]
        }
        for obj in record["objects"]:
            row = saved(record, obj)
            key = record["sensor_timestamp"], obj["track_id"]
            if key in fresh:
                data[f"archived_v3_{names[fresh[key]]}_fresh"].append(row)
            for prop, label in names.items():
                if obj["track_id"] in coverage["summary"][prop]["retained_ids"] and obj[
                    "diagnostics"
                ].get("static_prop_policy"):
                    data[f"archived_v3_{label}_retained"].append(row)
            if obj["track_id"] in humans:
                data["archived_v3_verified_human"].append(row)
    if hashes[str(source)] != mission["source_hashes"][str(source)]:
        raise ValueError("archived V3 source differs from original mission export")
    for prop, label in names.items():
        for phase, count in (
            ("fresh", "fresh_visual_rows"),
            ("retained", "retained_rows"),
        ):
            if (
                len(data[f"archived_v3_{label}_{phase}"])
                != coverage["summary"][prop][count]
            ):
                raise ValueError(
                    "archived V3 evaluation selection differs from original audit"
                )
    return dict(datasets=data, source_sha256=hashes)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--mission-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output
    if OUT.exists():
        parser.error("use a fresh report folder; preserve earlier response studies")
    OUT.mkdir(parents=True)
    payload = extract_native_rows(args.paired_manifest, args.mission_manifest)
    data = payload["datasets"]
    results = {}
    selected_rows = {}
    for key, rows in data.items():
        values = [assess(r, human=key.endswith("verified_human")) for r in rows]
        selected_rows[key] = [
            dict(
                frame=r["frame"],
                source=r["source"],
                track=r["object"]["track_id"],
                assessment=v.as_dict(),
            )
            for r, v in zip(rows, values)
        ]
        results[key] = dict(
            baseline=describe([v.original_phi for v in values]),
            candidate=describe([v.phi for v in values]),
            physical=describe([v.physical_phi for v in values]),
            eligible_count=sum(v.eligible for v in values),
            eligible_scores=describe([v.phi for v in values if v.eligible]),
            factors={
                n: describe(
                    [
                        next(
                            (f.score for f in v.factors if f.factor == n),
                            r["object"]["cra_control"]["cra"]["factors"][n][
                                "conservative_risk"
                            ],
                        )
                        for r, v in zip(rows, values)
                    ]
                )
                for n in ("thermal", "contact", "potential", "kinetic")
            },
            supported={
                n: sum(
                    next((f.supported for f in v.factors if f.factor == n), False)
                    for v in values
                )
                for n in ("thermal", "contact", "potential", "kinetic")
            },
            contact_context_prior_count=sum(
                any(
                    f.factor == "contact"
                    and f.contact_context_basis
                    == "configured_upper_interaction_prior_for_unavailable_context"
                    for f in v.factors
                )
                for v in values
            ),
            raw_proxy_rejection_count=sum(
                any(f.contact_proxy_rejections for f in v.factors) for v in values
            ),
            human_floor_uplift_count=sum(v.phi > v.physical_phi for v in values),
        )
    sweep = []
    for power, context, gain, scale, weight in itertools.product(
        (1.0, 2.0), (0.5, 0.75, 1.0), (0.0, 0.25), (20.0, 50.0, 100.0), (1.0, 1.5)
    ):
        config = RestaurantCalibrationConfig(
            enabled=True,
            contact_power=power,
            context_floor=context,
            uncertainty_gain=gain,
            kinetic_energy_scale_j=scale,
            weights=RestaurantFactorWeights(kinetic=weight),
        )
        values = {
            k: np.array([assess(r, config).phi for r in data[k]])
            for k in (
                "calibration_bin_parked",
                "calibration_crate_parked",
                "calibration_bin_live",
                "calibration_crate_live",
            )
        }
        b, c = values["calibration_bin_live"], values["calibration_crate_live"]
        sweep.append(
            dict(
                config=asdict(config),
                short_trial_statistics={k: describe(v) for k, v in values.items()},
                pairwise_order_fraction=float((c[:, None] > b[None, :]).mean()),
                median_gap=float(np.median(c) - np.median(b)),
            )
        )
    summary = dict(
        scope="Offline candidate-policy replay of recorded original V3 observations. No new wheels, controller acceptance or causal trajectory comparison.",
        selection="Small interpretable engineering grid on the isolated short trials and mathematical responses, with archived V3 human response informing the 50 J kinetic reference. Original V3 is a separate archived response check, not a blind holdout; no object label enters the scoring formula.",
        source_sha256=payload["source_sha256"],
        config=asdict(CONFIG),
        datasets=results,
        input_manifests_sha256={
            str(p): digest(p) for p in (args.paired_manifest, args.mission_manifest)
        },
        limitations=[
            "One driven short trial per prop; correlated field rows are not independent repetitions.",
            "Bin near edge is 21.4 cm farther from the route; this is not a causal risk-only controller comparison.",
            "Potential and thermal remain explicit .25 priors in these prop records.",
            "Rough mass and 2D outline proxy are uncalibrated; fresh-view outline outliers persist.",
            "Verified human protection .8 is a separate engineering prior, not evidence of sharpness or measured injury probability.",
            "The configured unknown-object policy remains unchanged; these targeted statistics describe qualified objects only.",
            "Kinetic upper energy includes uncertainty bounds; missing motion still uses a prior and unavailable contact context uses an upper interaction prior.",
        ],
        parameter_rationale={
            "contact_power": "Square preserves 0 and 1 and attenuates weak silhouette cues continuously; it is not learned sharpness calibration.",
            "context_floor": "A .75 floor keeps intrinsic consequences visible for nearby off-chord objects, with .25 continuous route modulation.",
            "uncertainty_gain": "Proportional reserve at most25 percent replaces a flat quarter-floor for supported cues; missing factors retain priors.",
            "kinetic_scale": "50 J is a disclosed engineering reference selected with archived V3 human response, not an injury threshold or independently validated optimum; short static-object upper energies mostly remain below1J.",
            "kinetic_weight": "1.5 positive noisy-OR exponent emphasizes motion without using percentage weights that dilute a strong single factor.",
            "potential": "100 J unchanged because the calibration pair provides no supported fall-energy evidence.",
        },
    )
    (OUT / "selected_candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (OUT / "selected_candidate_rows.json").write_text(
        json.dumps(selected_rows, indent=2) + "\n"
    )
    (OUT / "production_api_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")
    plot(results, selected_rows)
    summary["output_sha256"] = {
        name: digest(OUT / name)
        for name in (
            "selected_candidate_rows.json",
            "production_api_sweep.json",
            "calibration_response_study.png",
        )
    }
    (OUT / "selected_candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: dict(
                    baseline=v["baseline"]["median"],
                    candidate=v["candidate"]["median"],
                    maximum=v["candidate"]["maximum"],
                    recovered_context=v["contact_context_prior_count"],
                )
                for k, v in results.items()
            },
            indent=2,
        )
    )


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def archived_response_keys(data):
    """Read historical report keys without changing archived JSON or its values."""
    result = {}
    for key, value in data.items():
        renamed = (
            "archived_v3_" + key.removeprefix("heldout_")
            if key.startswith("heldout_")
            else key
        )
        if renamed in result:
            raise ValueError("ambiguous legacy/current archived response dataset key")
        result[renamed] = value
    return result


def plot(results, rows):
    results, rows = archived_response_keys(results), archived_response_keys(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    groups = [
        ("calibration_bin_live", "Short bin"),
        ("calibration_crate_live", "Short crate"),
        ("archived_v3_bin_retained", "V3 bin"),
        ("archived_v3_crate_retained", "V3 crate"),
        ("archived_v3_verified_human", "V3 human"),
    ]
    ax = axes[0, 0]
    for offset, field, color, label in (
        (-0.12, "baseline", "#777777", "Recorded policy"),
        (0.12, "candidate", "#2166ac", "Candidate replay"),
    ):
        med = [results[k][field]["median"] for k, _ in groups]
        low = [med[i] - results[k][field]["q10"] for i, (k, _) in enumerate(groups)]
        high = [results[k][field]["q90"] - med[i] for i, (k, _) in enumerate(groups)]
        ax.errorbar(
            np.arange(len(groups)) + offset,
            med,
            yerr=[low, high],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    ax.axhline(
        0.68359375, color="#b89b38", ls=":", label="Unassessed default unchanged"
    )
    ax.set_xticks(np.arange(len(groups)), [label for _, label in groups])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("CRA phi")
    ax.set_title("Median and 10–90% range (correlated observations)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for key, color, label in (
        ("archived_v3_bin_retained", "#298c55", "Bin"),
        ("archived_v3_crate_retained", "#aa5d20", "Crate"),
    ):
        v = rows[key]
        ax.plot(
            [r["source"] for r in v],
            [r["assessment"]["phi"] for r in v],
            ".",
            ms=3,
            color=color,
            label=label,
        )
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Original LiDAR source time (s)")
    ax.set_ylabel("Candidate replay phi")
    ax.set_title("Separate V3 retained-evidence response; gaps remain gaps")
    ax.legend()
    ax = axes[1, 0]
    cue = np.linspace(0, 1, 201)
    old = 0.5 * cue * 0.15 + 0.5 * 0.25
    new = np.minimum(1, cue**2 * (0.75 + 0.25 * 0.15) * (1 + 0.25 * 0.5))
    ax.plot(cue, old, label="Legacy: route .15, confidence .5", color="#777777")
    ax.plot(cue, new, label="Candidate: same upper cue/context", color="#2166ac")
    ax.set_xlabel("Upper visual contact cue (not physical sharpness)")
    ax.set_ylabel("Contact factor score")
    ax.set_ylim(0, 1.03)
    ax.set_title("Supported contact response")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    energy = np.linspace(0, 100, 400)
    for scale in (20, 50, 100):
        q = np.minimum(1, np.minimum(1, energy / scale) * 0.7875 * 1.125)
        score = 0.25 + 0.75 * (1 - (1 - q) ** 1.5)
        ax.plot(
            energy, score, label=f"{scale} J reference", lw=2.5 if scale == 50 else 1
        )
    ax.set_xlabel("Upper kinetic energy (J)")
    ax.set_ylabel("Phi: one supported factor + prior .25")
    ax.set_ylim(0, 1.03)
    ax.set_title("Kinetic sensitivity; exponent 1.5, context .15, confidence .5")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Restaurant CRA sensitivity study — archived response, not a new navigation trial",
        fontsize=14,
    )
    fig.savefig(OUT / "calibration_response_study.png", dpi=170)


if __name__ == "__main__":
    main()
