"""One background sensor-inference job, with acquisition-time freshness checks.

Only immutable copied sensor inputs belong in the worker; Isaac/USD APIs and
wheel commands stay on the simulation thread. No observation stop is issued.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class InferenceOutcome:
    frame_id: int
    sensor_timestamp_seconds: float
    value: object | None
    rejection_reason: str | None
    wall_seconds: float

    @property
    def accepted(self):
        return self.rejection_reason is None


class LiveInferenceRunner:
    """Bounded single-job scheduling: no accumulating backlog or retimed results."""

    def __init__(self, function, *, maximum_result_age_seconds=0.35):
        if not callable(function):
            raise TypeError("inference function must be callable")
        if (
            not math.isfinite(maximum_result_age_seconds)
            or maximum_result_age_seconds <= 0
        ):
            raise ValueError("maximum result age must be positive")
        self.function = function
        self.maximum_age = float(maximum_result_age_seconds)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cra-perception"
        )
        self.pending = None
        self.last_frame = -1
        self.last_time = -1.0
        self.closed = False

    @property
    def busy(self):
        """Whether the single slot still owns a pending or unconsumed job."""
        return self.pending is not None

    def submit(self, *, frame_id, sensor_timestamp_seconds, packet):
        if self.closed:
            raise RuntimeError("inference runner is closed")
        if (
            isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or frame_id <= self.last_frame
            or isinstance(sensor_timestamp_seconds, bool)
            or not math.isfinite(sensor_timestamp_seconds)
            or sensor_timestamp_seconds < 0
            or sensor_timestamp_seconds <= self.last_time
        ):
            raise ValueError(
                "sensor frame IDs and timestamps must be strictly increasing"
            )
        if self.pending is not None:
            return False
        self.last_frame, self.last_time = frame_id, float(sensor_timestamp_seconds)
        started = time.perf_counter()
        future = self.executor.submit(self.function, packet)
        self.pending = (frame_id, float(sensor_timestamp_seconds), started, future)
        return True

    def poll(self, *, current_sensor_time_seconds):
        if (
            not math.isfinite(current_sensor_time_seconds)
            or current_sensor_time_seconds < 0
        ):
            raise ValueError("current sensor time must be finite and nonnegative")
        if self.pending is None or not self.pending[3].done():
            return None
        frame, source_time, started, future = self.pending
        self.pending = None
        value, reason = None, None
        try:
            value = future.result()
            if value is None:
                reason = "sensor inference returned no result"
        except Exception as error:
            reason = f"sensor inference failed: {type(error).__name__}: {error}"
        if current_sensor_time_seconds < source_time - 1e-9:
            reason = "inference result is future evidence"
        elif current_sensor_time_seconds - source_time > self.maximum_age:
            reason = "inference result is stale at acquisition-time age"
        return InferenceOutcome(
            frame,
            source_time,
            value if reason is None else None,
            reason,
            time.perf_counter() - started,
        )

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
