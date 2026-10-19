"""Optional worker profiling; never changes an input, result or source epoch."""

import cProfile
import math
import os
from pathlib import Path
import sys
import time
import warnings


def profile_call(function, *args, profile_path=None, minimum_seconds=3.0, **kwargs):
    if profile_path is None:
        return function(*args, **kwargs)
    if not math.isfinite(minimum_seconds) or minimum_seconds < 0:
        raise ValueError("finite nonnegative profile threshold required")
    profile = cProfile.Profile()
    started = time.perf_counter()
    try:
        return profile.runcall(function, *args, **kwargs)
    finally:
        if time.perf_counter() - started >= minimum_seconds:
            path = Path(profile_path)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                profile.dump_stats(str(path))
            except OSError as error:
                warnings.warn(f"runtime profile was not saved: {error}", RuntimeWarning)


def numeric_runtime_contract():
    modules = {}
    for name in ("numpy", "scipy", "osqp", "torch", "torchvision"):
        module = sys.modules.get(name)
        modules[name] = (
            None
            if module is None
            else {
                "path": getattr(module, "__file__", None),
                "version": getattr(module, "__version__", None),
            }
        )
    try:
        from threadpoolctl import threadpool_info

        pools = {"available": True, "pools": threadpool_info()}
    except ImportError:
        pools = {
            "available": False,
            "reason": "optional threadpoolctl is not installed; no packages installed",
        }
    return {
        "modules": modules,
        "thread_pools": pools,
        "logical_cpu_count": os.cpu_count(),
        "thread_count_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "numpy_build_configuration": repr(
            getattr(
                getattr(sys.modules.get("numpy"), "__config__", None), "CONFIG", None
            )
        ),
        "scope": "read-only loaded numeric-library contract; no thread limits changed",
    }
