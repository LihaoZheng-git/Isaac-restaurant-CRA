# Restaurant service experiment

Carter starts with the first meal, delivers it to the left table, returns to the service bar, and takes the second meal to the right table. All three phases share one Isaac Sim session. The bin and walking diner appear on the left delivery. The running child and the server pushing a trolley cross the right delivery route.

## Inputs

1. Follow the [environment setup](../README.md) and activate the Python environment.
2. Run `python -m experiment.scene.prepare` from the repository root to prepare the [restaurant scene](scene/README.md).
3. For a robot run, download the two public weights with `python -m experiment.models` and obtain the two CRA weights listed in the [model guide](models/README.md). Viewing the prepared scene in Isaac Sim does not require these weights.

## Run

From the repository root in PowerShell, set `$isaac` to your Isaac Sim launcher. Use `--check` to verify the inputs or `--output` to run the headless DirectX 12 experiment.

```powershell
$scene = 'experiment/scene/prepared'
$isaac = 'C:/isaacsim/python.bat'

python -m experiment.run --scene-root $scene --isaac-python $isaac --check
python -m experiment.run --scene-root $scene --isaac-python $isaac --output outputs/restaurant-service
```

Use `--model-root` if your weights are in another folder. Each run saves camera recordings, sensor data, and results under `--output`.
