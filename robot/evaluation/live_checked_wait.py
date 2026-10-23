"""Opt-in live field admission; rejected asynchronous proposals never become history."""

import copy
import math

import numpy as np

from robot.safety.checked_wait_transition import assess_checked_wait_transition


def packet_state_in_field(packet, field):
    """Transform the acquisition pose into the fixed field anchor, including yaw."""
    c, s = math.cos(field.anchor_yaw), math.sin(field.anchor_yaw)
    delta = np.asarray(packet.robot_position_world_xy) - field.anchor_position_world_xy
    yaw = packet.robot_yaw - field.anchor_yaw
    return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1],
                     math.atan2(math.sin(yaw), math.cos(yaw))])


def check_wait(control, state, field):
    return assess_checked_wait_transition(
        control.mpc, state, field,
        lambda: control.filter_command((0., 0.), state, field),
    )


class StagedLivePerception:
    """Single-job adapter for the existing transactional all-object worker.

    The worker already clones tracker/evidence state. A shallow staging shell
    isolates its final attribute assignments until source-age, current-pose and
    coverage admission on the simulation thread. Model image caches are shared;
    they are not accepted tracking/evidence state. No concurrent jobs are allowed.
    """

    def __init__(self, worker):
        if worker.persistent_properties is None or worker.encounter_motion is None:
            raise ValueError("staged checked waiting requires encounter object-map mode")
        self.worker = worker
        self.pending = None

    def __call__(self, packet):
        if self.pending is not None:
            raise RuntimeError("previous perception proposal has not been consumed")
        staged = copy.copy(self.worker)
        try:
            result = staged(packet)
        except Exception:
            # Preserve only the existing explicitly unpublished temporal-gap
            # bootstrap. It supplies no control field or accepted evidence state.
            baseline = getattr(staged, "temporal_recovery_baseline", None)
            if baseline is not None:
                self.worker.temporal_recovery_baseline = baseline
            raise
        self.pending = (result, staged)
        return result

    def accept(self, result):
        if self.pending is None or self.pending[0] is not result:
            raise ValueError("only the current original proposal may be accepted")
        self.worker.__dict__.update(self.pending[1].__dict__)
        self.pending = None

    def discard(self):
        self.pending = None
