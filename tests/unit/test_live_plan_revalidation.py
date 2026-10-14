import numpy as np
import pytest
from test_safe_sage_cbf_filter import make_fields
from test_safe_sage_unicycle_mpc import make_mpc


def test_delayed_plan_requires_exact_current_field_validation_and_actuator_bounds():
    mpc = make_mpc(safety_substeps_per_interval=4)
    commands = np.zeros((mpc.config.horizon_steps, 2))

    def check(value, unsafe=False):
        configuration, h, guidance = make_fields(
            safety_value=value,
            gradient=(0.0, 0.0, 0.0),
            guidance=(0.0, 0.0),
            unsafe_yaw_indices=(0, 1, 2, 3) if unsafe else (),
        )
        return mpc.revalidate_commands(
            np.zeros(3),
            commands,
            configuration_space=configuration,
            h_full=h,
            guidance_fields=guidance,
        )

    assert check(1.0)
    assert not check(-1.0)
    assert not check(1.0, True)
    commands[0, 0] = 10.0
    assert not check(1.0)


def test_malformed_async_horizon_is_rejected():
    mpc = make_mpc()
    config, h, guidance = make_fields()
    with pytest.raises(ValueError, match="complete horizon"):
        mpc.revalidate_commands(
            np.zeros(3),
            np.zeros((2, 2)),
            configuration_space=config,
            h_full=h,
            guidance_fields=guidance,
        )
