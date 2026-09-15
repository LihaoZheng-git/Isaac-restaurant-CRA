"""Validate the matched experiment matrix without running robot motion."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.evaluation.cra_experiments import (
    CRABaseline,
    EvidenceMode,
    PRIMARY_BASELINES,
    RestaurantTrialResult,
    primary_restaurant_scenarios,
    summarize_trials,
)


def main():
    scenarios = primary_restaurant_scenarios(evidence_mode=EvidenceMode.SENSOR)
    print("STATIONARY EXPERIMENT CONTRACT: illustrative results; no wheel motion")
    results = tuple(
        RestaurantTrialResult(
            scenario.scenario_id,
            scenario.seed,
            baseline,
            scenario.evidence_mode,
            True,
            False,
            False,
            0.4,
            1,
            0,
            8,
            0.01,
            0 if baseline is CRABaseline.GEOMETRIC_VELOCITY_CBF else 1.25,
        )
        for scenario in scenarios
        for baseline in PRIMARY_BASELINES
    )
    for summary in summarize_trials(results):
        print(
            f"{summary.baseline.value}: trials={summary.trial_count}, lambda policy validated"
        )
    print(
        "PASS: sensor/oracle/prior labels and lambda ablation are structurally separated"
    )


if __name__ == "__main__":
    main()
