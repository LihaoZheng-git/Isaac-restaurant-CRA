# Restaurant service experiment

[Watch the full restaurant demo](../project%20webpage/demo%20result/restaurant.mp4)

Carter starts with the first meal, delivers it to the left table, returns to the service bar, and takes the second meal to the right table. All three phases share one Isaac Sim session. The bin and walking diner appear on the left delivery. The running child and the server pushing a trolley cross the right delivery route.

## Inputs

1. Follow the [environment setup](../README.md) and activate the Python environment with the vision models.
2. Prepare the [restaurant scene](scene/README.md). The commands below use `experiment/scene/prepared`.
3. Supply the four checkpoint files in one model directory. The original run's checksums are in [recorded-run.json](recorded-run.json). Obtain the project-trained ABO mass and PACO contact checkpoints from the project authors.

| Filename in the model directory | Model | Optional environment variable for an existing file |
| --- | --- | --- |
| `sam2.1_t.pt` | Ultralytics SAM 2.1 tiny | `SAM2_CHECKPOINT` |
| `visual-mass.pt` | CRA visual mass estimator trained on ABO | `VISUAL_MASS_CHECKPOINT` |
| `contact-segmentation.pt` | CRA exposed-edge segmenter trained on PACO | `CONTACT_SEGMENTATION_CHECKPOINT` |
| `yolo11n-seg.pt` | Ultralytics YOLO11n segmentation for people | `PERSON_SEGMENTATION_CHECKPOINT` |

## Run

From the repository root in PowerShell, point `$models` and `$isaac` at your local files. The first command checks the recipe and fingerprints without starting the simulator. The second starts the headless DirectX 12 experiment and writes to a new output directory.

```powershell
$scene = 'experiment/scene/prepared'
$models = 'D:/CRA/models'
$isaac = 'C:/isaacsim/python.bat'

python -m experiment.run --scene-root $scene --model-root $models --isaac-python $isaac --check
python -m experiment.run --scene-root $scene --model-root $models --isaac-python $isaac --output outputs/restaurant-service
```

The [mission recipe](configs/mission.json) selects the three phase configurations and enables the animated cast and LiDAR-guided prop masks. Native camera recordings, sensor data, per-phase reports, and the mission result are written under the output directory. The [recorded v2 run](recorded-run.json) identifies the controller revision and the outcome of each phase used for the published video.
