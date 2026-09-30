"""Print the reusable restaurant scenario catalog without launching Isaac Sim."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.scene.restaurant_risk_scenarios import (
    restaurant_risk_scenario_manifest,
)


def main() -> None:
    payload = restaurant_risk_scenario_manifest()
    print("RESTAURANT RISK SCENARIO CATALOG: authored oracle definitions")
    print("robot_motion=none; runtime_sensor_claims=none")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
