"""Replay score candidates on recorded ownership; no counterfactual dynamics."""

from collections import Counter
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robot.evaluation.restaurant_calibration_runtime import (
    object_is_eligible,
    restaurant_calibration_preset,
    verified_human_ids,
)
from robot.risk.restaurant_calibration import evaluate_restaurant_calibration


def score(obj, record, humans, config, static_property_age=120.0):
    d, key = obj["diagnostics"], obj["track_id"]
    eligible = object_is_eligible(d) or (
        key in humans and d.get("policy_return_count", 0) > 0
    )
    persistent = (
        d.get("property_evidence", {}).get("contact", {}).get("mode")
        == "persistence_prior"
    )
    max_age = (
        static_property_age
        if d.get("static_prop_policy")
        else record["diagnostics"]["maximum_property_age_seconds"]
        if persistent
        else 1.0
    )
    return evaluate_restaurant_calibration(
        obj["cra_control"]["cra"],
        d["factor_scores"],
        config,
        eligible=eligible,
        verified_human=key in humans,
        source_energy_scales_j={"potential": 100.0, "kinetic": 100.0},
        contact_proxy=d.get("contact_proxy"),
        contact_interaction=obj["attributes"]["motion"]["route_crossing_likelihood"],
        timestamp_seconds=obj["attributes"]["timestamp_seconds"],
        maximum_contact_age_seconds=max_age,
    )


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_field(record, expected):
    """Exact archived identity and clocks, never nearest-time matching."""
    if (
        record["sensor_frame_id"] != expected["field_id"]
        or record["sensor_timestamp"] != expected["source_time"]
        or record["commit_simulation_time"] != expected["commit_time"]
    ):
        raise ValueError("field identity or original clocks differ from manifest")
    if record["commit_simulation_time"] < record["sensor_timestamp"]:
        raise ValueError("field was committed before its source observation")


def replay_policies(record, individual):
    """Apply candidate object values while preserving external/geometry floors."""
    if (record["diagnostics"].get("carried_object_groups") or {}).get("confirmed_groups"):
        raise ValueError("carried group replay is outside this archived fixture scope")
    envelopes = {}
    for row in record["diagnostics"]["human_body_envelope_priors"]:
        if row.get("applied_to_occupancy"):
            child, parent = row["envelope_track_id"], row["track_id"]
            if child in envelopes or parent not in individual:
                raise ValueError("invalid envelope to parent mapping")
            envelopes[child] = parent
    policies = []
    seen = set()
    for old in record["diagnostics"]["applied_field_policies"]:
        key = old["track_or_group_id"]
        if key in seen:
            raise ValueError("duplicate final policy ID")
        seen.add(key)
        original, cra = float(old["applied_field_risk"]), float(old["cra_phi"])
        if not (0 <= original <= 1 and 0 <= cra <= 1):
            raise ValueError("invalid original policy risk")
        candidate = individual[key].phi if key in individual else cra
        floor = original if original > cra + 1e-12 else None
        parent = envelopes.get(key)
        if parent is not None:
            candidate = individual[parent].phi
            floor = original  # Existing body uncertainty prior never reduced.
        applied = max(candidate, floor) if floor is not None else candidate
        if key not in individual and parent is None:
            applied = original
        policies.append(
            dict(
                track_id=key,
                original_cra_phi=cra,
                original_applied_phi=original,
                candidate_individual_phi=candidate,
                preserved_external_floor=floor,
                candidate_current_phi=applied,
                candidate_prior_geometry_phi=original,
                envelope_parent_track_id=parent,
                original_boundary_flux=old["boundary_flux"],
                candidate_boundary_flux=-(0.5 + 1.2 * applied),
                lambda_value=old["lambda"],
            )
        )
    if not set(envelopes) <= seen:
        raise ValueError("applied human envelope has no final policy")
    return policies


def records(path):
    with path.open("rb") as stream:
        if path.suffix == ".jsonl":
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            yield from json.load(stream)


def export(manifest_path, output, *, preset="kinetic", static_property_age=120.0):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    source_directory = Path(manifest["source_directory"])
    sources = [
        source_directory / name
        for name in ("stationary-fields.json", "live-fields.jsonl")
    ]
    if (
        output.resolve() == source_directory.resolve()
        or source_directory.resolve() in output.resolve().parents
    ):
        raise ValueError("candidate output must not modify the native archive")
    if not 0 < static_property_age <= 120.0:
        raise ValueError(
            "archived static property age must be positive and no longer than original120s"
        )
    source_hashes = {}
    for source in sources:
        expected_hash = manifest["source_hashes"][str(source)]
        actual_hash = digest(source)
        if actual_hash != expected_hash:
            raise ValueError(f"original source hash changed: {source.name}")
        source_hashes[str(source)] = actual_hash
    expected = {r["field_id"]: r for r in manifest["fields"]}
    if len(expected) != len(manifest["fields"]):
        raise ValueError("duplicate exported field ID")
    output.mkdir(parents=True, exist_ok=True)
    config = restaurant_calibration_preset(preset)
    counts, seen = Counter(), set()
    destination = output / "candidate_policy_fields.jsonl"
    with destination.open("w", encoding="utf-8") as target:
        for r in (row for source in sources for row in records(source)):
            key = r["sensor_frame_id"]
            if key not in expected or key in seen:
                raise ValueError("native field is absent or duplicated in manifest")
            validate_field(r, expected[key])
            seen.add(key)
            humans = verified_human_ids(
                r["objects"],
                r["diagnostics"]["human_body_envelope_priors"],
                r["sensor_timestamp"],
            )
            object_rows = []
            individual = {}
            for obj in r["objects"]:
                result = score(obj, r, humans, config, static_property_age)
                key = obj["track_id"]
                if key in individual:
                    raise ValueError("duplicate object ID")
                individual[key] = result
                row = dict(
                    track_id=key,
                    policy_return_count=obj["diagnostics"]["policy_return_count"],
                    eligible=result.eligible,
                    verified_human=result.verified_human,
                    reason=result.reason,
                    original_phi=result.original_phi,
                    physical_phi=result.physical_phi,
                    candidate_phi=result.phi,
                    prior_baseline=result.prior_baseline,
                    factors={
                        f.factor: dict(
                            score=f.score,
                            supported=f.supported,
                            mode=f.mode,
                            contact_context_basis=f.contact_context_basis,
                            contact_proxy_rejections=f.contact_proxy_rejections,
                        )
                        for f in result.factors
                    },
                )
                object_rows.append(row)
                counts["objects"] += 1
                counts["eligible_objects"] += result.eligible
                counts["verified_human_objects"] += result.verified_human
                counts["changed_objects"] += (
                    abs(result.phi - result.original_phi) > 1e-12
                )
                counts["zero_return_objects"] += row["policy_return_count"] == 0
                if not result.eligible and not result.verified_human:
                    assert result.phi == result.original_phi
                    counts["ordinary_unchanged_objects"] += 1
            policies = replay_policies(r, individual)
            for policy in policies:
                counts["policy_rows"] += 1
                counts["external_floors_preserved"] += (
                    policy["preserved_external_floor"] is not None
                )
                counts["human_envelope_parent_mappings"] += (
                    policy["envelope_parent_track_id"] is not None
                )
                counts["changed_current_policies"] += (
                    abs(
                        policy["candidate_current_phi"] - policy["original_applied_phi"]
                    )
                    > 1e-12
                )
            field_payload = dict(
                sensor_frame_id=r["sensor_frame_id"],
                sensor_timestamp=r["sensor_timestamp"],
                commit_simulation_time=r["commit_simulation_time"],
                objects=object_rows,
                policies=policies,
            )
            target.write(json.dumps(field_payload, separators=(",", ":")) + "\n")
            counts["fields"] += 1
    if seen != set(expected):
        raise ValueError("export does not cover every original manifest field")
    for source in sources:
        if digest(source) != source_hashes[str(source)]:
            raise ValueError("source changed while scoring")
    summary = dict(
        source_directory=str(source_directory),
        config=asdict(config),
        counts=dict(counts),
        scope="Archived ownership + candidate current-object scoring; original trajectory and prior geometry risk preserved. Not a reconstructed controller rollout.",
        external_floor_rule="If recorded final applied risk exceeds recorded CRA phi, retain that external floor. Ordinary rows remain exact.",
        prior_geometry_rule="Use candidate_prior_geometry_phi for hollow prior/retained placements; only current observed returns use candidate_current_phi.",
        static_property_age_seconds=static_property_age,
        static_property_age_source="0047241 static_prop_runtime / StaticPropEvidenceConfig; generic retained age remains original row setting.",
        source_hashes=source_hashes,
        manifest_sha256=digest(manifest_path),
        fields_file=destination.name,
        fields_sha256=digest(destination),
        original_manifest=str(manifest_path.resolve()),
        source_hashes_verified_after_export=True,
    )
    (output / "candidate_policy_fields_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("equal", "kinetic"), default="kinetic")
    parser.add_argument("--static-property-age", type=float, default=120.0)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.manifest,
                args.output,
                preset=args.preset,
                static_property_age=args.static_property_age,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
