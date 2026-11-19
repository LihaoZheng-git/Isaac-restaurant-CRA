# Model weights

The full restaurant mission uses four checkpoints. You can prepare and open the restaurant scene before installing them.

1. From the repository root, download the two public checkpoints into this folder.

   ```sh
   python -m experiment.models
   ```

   This downloads and verifies [Meta SAM 2.1 Hiera Tiny](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) and [Ultralytics YOLO11n segmentation](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt).

2. [Request access to the two CRA checkpoints](https://github.com/LihaoZheng-git/Isaac-restaurant-CRA/issues/new?title=Request%20CRA%20v2%20model%20checkpoints). Place the ABO visual mass estimator at `experiment/models/visual-mass.pt` and the PACO contact segmenter at `experiment/models/contact-segmentation.pt`.

3. Verify all four files against the [recorded demo hashes](../recorded-run.json) before starting Isaac Sim.

   ```sh
   python -m experiment.models --check
   ```

If your checkpoints are together in another folder, pass `--model-root` to both this command and the [experiment launcher](../README.md).
