"""Authored fixture separation and stationary-only mounting constraints."""

import ast
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import pytest

from robot.scene.restaurant_container_bench import CONTAINER_PATH, author_service_container
from robot.scene.restaurant_demo_build import DEMO_ROOT, author_demo_props
from test_restaurant_demo_build import _stage


@pytest.mark.parametrize("shape", ["angular", "rounded"])
def test_matched_container_stays_ground_supported_and_never_supplies_factors(shape):
    stage = _stage()
    author_demo_props(stage)
    before = stage.GetRootLayer().ExportToString()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        row = author_service_container(stage, shape)
    assert stage.GetRootLayer().ExportToString() == before
    assert not stage.GetPrimAtPath(DEMO_ROOT + "/RoundedBin").IsActive()
    np.testing.assert_allclose(row["bounds_minimum"], [27.99, 5.94, 0.], atol=1e-6)
    np.testing.assert_allclose(row["bounds_maximum"], [28.31, 6.26, .48], atol=1e-6)
    assert row["available_as_physical_factor_input"] is False
    for prim in Usd.PrimRange(stage.GetPrimAtPath(CONTAINER_PATH)):
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert not prim.HasAttribute("phi")
        assert not prim.HasAttribute("semantic:class")
    with pytest.raises(ValueError, match="active original"):
        author_service_container(stage, shape)


def test_prop_isolation_disables_only_the_separate_layers_server():
    stage = _stage()
    author_demo_props(stage)
    server = UsdGeom.Xform.Define(stage, "/World/CRAServiceActors/Server").GetPrim()
    before = stage.GetRootLayer().ExportToString()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        author_service_container(stage, "angular", props_only=True)
    assert server.IsValid() and not server.IsActive()
    assert stage.GetRootLayer().ExportToString() == before
    assert stage.GetPrimAtPath(DEMO_ROOT + "/UtensilCaddy").IsActive()


def _parse_mount(argv, monkeypatch):
    """Exercise the real parser and its guards without importing Isaac."""
    script = Path("scripts/demos/cra_live_object_isaac_demo.py")
    tree = ast.parse(script.read_text(encoding="utf-8"))
    # Stop before output directories, environment changes or Isaac imports.
    cutoff = next(i for i, node in enumerate(tree.body)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and ast.unparse(node.value.func) == "args.output.mkdir")
    namespace = {"__file__": str(script.resolve()), "__name__": "cli_test"}
    monkeypatch.setattr("sys.argv", [str(script), "--output", "unused", *argv])
    exec(compile(ast.Module(tree.body[:cutoff], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["args"]


BASE_ARGS = ["--stationary-prop-camera", "--human-camera-raise-metres", ".6",
             "--restaurant-scene", "fixture.usda", "--restaurant-view", "demo-container",
             "--persistent-object-risk-map", "--static-occupancy-memory-prior",
             "--restaurant-layout-survey", "survey.json"]


@pytest.mark.parametrize("extra", [["--motion"], ["--restaurant-view", "bowl"],
                                  ["--restaurant-human", "standing"], ["--human-camera-raise-metres", ".61"]])
def test_prop_camera_cannot_enable_general_or_moving_mount(extra, monkeypatch):
    with pytest.raises(SystemExit):
        _parse_mount([*BASE_ARGS, *extra], monkeypatch)


def test_real_stationary_prop_mount_arguments_pass(monkeypatch):
    args = _parse_mount(BASE_ARGS, monkeypatch)
    assert args.human_camera_raise_metres == .6 and not args.motion


def test_extent_retry_cli_requires_context_opt_in(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_mount([*BASE_ARGS, "--sam-track-extent-retry"], monkeypatch)
    args = _parse_mount([*BASE_ARGS, "--sam-track-extent-retry", "--sam-observed-context-box"], monkeypatch)
    assert args.sam_track_extent_retry and args.sam_observed_context_box


@pytest.mark.parametrize('extra', [[], ['--motion'], ['--restaurant-view', 'pot'],
                                  ['--stationary-prop-standoff-metres', 'nan'],
                                  ['--stationary-prop-standoff-metres', '.79'],
                                  ['--stationary-prop-standoff-metres', '3.1']])
def test_prop_standoff_is_bounded_stationary_only(extra, monkeypatch):
    argv = [] if not extra else BASE_ARGS
    with pytest.raises(SystemExit):
        _parse_mount([*argv, '--stationary-prop-standoff-metres', '1.1', *extra], monkeypatch)


def test_prop_standoff_accepts_separate_near_and_far_surveys(monkeypatch):
    for distance in ('1.1', '2.5', '2.8'):
        args = _parse_mount([*BASE_ARGS, '--stationary-prop-standoff-metres', distance], monkeypatch)
        assert args.stationary_prop_standoff_metres == float(distance) and not args.motion


@pytest.mark.parametrize('angle', ['nan', '-1', '360'])
def test_prop_view_angle_rejects_invalid_values(angle, monkeypatch):
    with pytest.raises(SystemExit):
        _parse_mount([*BASE_ARGS, '--stationary-prop-standoff-metres', '2.5',
                      '--stationary-prop-view-yaw-degrees', angle], monkeypatch)


def test_prop_view_angle_needs_standoff_and_cannot_enable_wheels(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_mount([*BASE_ARGS, '--stationary-prop-view-yaw-degrees', '90'], monkeypatch)
    args = _parse_mount([*BASE_ARGS, '--stationary-prop-standoff-metres', '2.5',
                         '--stationary-prop-view-yaw-degrees', '90'], monkeypatch)
    assert args.stationary_prop_view_yaw_degrees == 90 and not args.motion
