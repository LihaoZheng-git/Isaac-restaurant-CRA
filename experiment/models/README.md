# Model weights

From the repository root, download the two public checkpoints into this folder:

```powershell
python -m experiment.models
```

The command downloads [Meta SAM 2.1 Hiera Tiny](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) as `sam2.1_t.pt` and [Ultralytics YOLO11n segmentation](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt) as `yolo11n-seg.pt`. It verifies both against the [recorded demo hashes](../recorded-run.json).

[Request the two CRA checkpoints from the authors](https://github.com/LihaoZheng-git/Isaac-restaurant-CRA/issues/new?title=Request%20CRA%20v2%20model%20checkpoints). Save the visual mass estimator as `visual-mass.pt` and the contact segmenter as `contact-segmentation.pt` in this folder.
