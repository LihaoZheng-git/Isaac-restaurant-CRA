"""Existing property-memory guarantees must also hold for verified groups."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_image_object_association import proposal
from test_image_object_runtime import observed, setup
from test_object_risk_map import scalar
from robot.perception.physical_attributes import EstimateStatus, MassAttributes


def prepared():
    runtime = setup()
    runtime.image_object_property_persistence = True
    mask = np.zeros((100, 100), bool)
    mask[35:45, 34:67] = True
    runtime.segmenter.image_proposals = lambda *a: (proposal(mask),)
    for i, t in enumerate((1., 1.1, 1.2), 1):
        result = runtime(observed(i, t))
    runtime.persistent_properties.freeze_properties()
    return runtime, group(result)


def group(result):
    return next(o for o in result.objects if o.diagnostics.get("image_object_association"))


def test_group_retains_original_mass_contact_but_motion_stays_live():
    runtime, original = prepared()
    runtime.mass_estimator = SimpleNamespace(estimate_instance=lambda *a, **kw:
        MassAttributes(scalar(9, "kg", kw["timestamp_seconds"])))
    current = replace(observed(4, 1.3), measured_robot_velocity_world_xy=(.2, 0.))
    later = group(runtime(current))
    assert later.attributes.track_id == original.attributes.track_id
    mass = later.attributes.mass.mass_kg
    assert mass.value == 2
    assert mass.status is EstimateStatus.PRIOR
    assert mass.timestamp_seconds == original.attributes.mass.mass_kg.timestamp_seconds
    assert mass.confidence < original.attributes.mass.mass_kg.confidence
    for name in ("mass", "contact"):
        detail = later.diagnostics["property_evidence"][name]
        assert detail["mode"] == "persistence_prior"
        assert detail["source_timestamp"] == original.diagnostics["property_evidence"][name]["source_timestamp"]
    assert later.diagnostics["factor_scores"]["kinetic"] != original.diagnostics["factor_scores"]["kinetic"]
    assert later.diagnostics["policy_return_count"] == 16


def test_group_loss_discards_memory_without_discarding_raw_occupancy():
    runtime, original = prepared()
    runtime.segmenter.image_proposals = lambda *a: ()
    lost = runtime(observed(4, 1.3))
    assert not any(o.diagnostics.get("image_object_association") for o in lost.objects)
    assert not runtime.image_object_state.property_memory.properties
    assert lost.field.all_obstacle_point_count == 16
    assert all(o.attributes.track_id != original.attributes.track_id for o in lost.objects)


def test_group_property_and_identity_state_are_transactional(monkeypatch):
    import robot.evaluation.multi_object_perception as mapping
    runtime, _ = prepared()
    original = runtime.image_object_state
    def fail(*a, **kw):
        raise RuntimeError("controlled field failure")
    monkeypatch.setattr(mapping, "build_safe_sage_field_pipeline", fail)
    with pytest.raises(RuntimeError, match="controlled field failure"):
        runtime(observed(4, 1.3))
    assert runtime.image_object_state is original
    assert original.property_memory.last_timestamp == pytest.approx(1.3)


def test_group_original_property_expiry_is_not_renewed_by_fresh_view():
    runtime, original = prepared()
    memory = runtime.image_object_state.property_memory
    memory.config = replace(memory.config, maximum_property_age_seconds=.05)
    runtime.mass_estimator = SimpleNamespace(estimate_instance=lambda *a, **kw:
        MassAttributes(scalar(9, "kg", kw["timestamp_seconds"])))
    later = group(runtime(observed(4, 1.3)))
    assert later.attributes.mass.mass_kg.value == 9
    assert later.attributes.mass.mass_kg.timestamp_seconds > original.attributes.mass.mass_kg.timestamp_seconds
    assert later.diagnostics["property_evidence"]["mass"]["mode"] == "fresh"


@pytest.mark.parametrize("change", [None, "legacy_motion", "no_memory", "short_gate", "human", "no_deskew"])
def test_approach_cli_keeps_legacy_and_physical_prerequisites(monkeypatch, change):
    path = Path("scripts/demos/cra_live_object_isaac_demo.py").resolve()
    command = json.loads(Path("configs/restaurant_crate_live_approach.json").read_text())["commands"]["static"]
    command += ["--motion", "--output", "outputs/unused-cli-test"]
    if change == "legacy_motion":
        command.remove("--image-object-approach")
    elif change == "no_memory":
        command.remove("--image-object-property-persistence")
    elif change == "short_gate":
        command[command.index("--stationary-frames") + 1] = "18"
    elif change == "human":
        command += ["--restaurant-human", "parallel"]
    elif change == "no_deskew":
        command += ["--disable-lidar-deskew"]
    monkeypatch.setattr("sys.argv", command)
    # Execute only the argument contract, before any output/Isaac initialization.
    code = compile(path.read_text().split("args.output.mkdir", 1)[0], str(path), "exec")
    namespace = {"__file__": str(path)}
    if change is None:
        exec(code, namespace)
        assert namespace["args"].image_object_approach
    else:
        with pytest.raises(SystemExit) as error:
            exec(code, namespace)
        assert error.value.code == 2
