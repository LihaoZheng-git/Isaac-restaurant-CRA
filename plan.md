# CRA Project Progress and Plan

## Work Completed

- Set up the basic restaurant environment in Isaac Sim.
- Integrated the Nova Carter robot, sensors, and differential-drive controller
  for basic perception and motion tests.
- Implemented and validated a simple geometric CBF safety-filter baseline using
  real LiDAR obstacle points.
- Completed the waypoint-navigation baseline design (planner) and nominal-control
  interface for restaurant navigation.

## Plan Ahead

- Add reusable restaurant scenarios for kinetic, contact, thermal, and
  potential risks.
- Add dynamic-object tracking and ground-truth CRA experiments.
- Design the continuous danger-score function and its visual-reasoning
  components.
- Map the danger score into a Poisson/Laplace-guided CBF safety filter.
- Compare geometric, class-aware, and CRA-aware safety-filter baselines.
- Complete simulation experiments, a small real-robot validation, and the
  research paper.

## Estimated Timeline

| Period | Target |
|---|---|
| August 1 - August 12 | Build reusable restaurant risk scenarios |
| August 13 - August 21 | Add dynamic tracking and ground-truth CRA inputs |
| August 22 - August 30 | Design the initial danger-score function and fine-tune |
| August 31 - September 8 | Implement the CRA-aware CBF prototype |
| September 9 - September 17 | Develop the visual-reasoning prototype |
| September 18 - September 26 | Integrate visual CRA with robot control |
| September 27 - October 5 | Run simulation experiments and ablations |
| October 6 - October 20 | Conduct a focused real-robot experiment |
| October 21 - November 1 | Final experiments, paper writing, and submission preparation |

