# CRA: Continuous Risk Assessment for Safety-critical Visual Navigation

[Project page](https://isaac-restaurant-cra.cra-demos.workers.dev/)

CRA combines camera and LiDAR observations to assess risk around the robot. The waypoint follower supplies a nominal velocity. The controller uses the local risk field to adjust that command before sending it to Carter's wheels.

![CRA pipeline from sensor observations and physical risk factors to navigation control](project%20webpage/media/pipeline.png)

The algorithm is in [robot/](robot/), including the [safety filter and MPC](robot/safety/). The restaurant mission and its configuration are in [experiment/](experiment/).

## Environment setup

1. Check your GPU, available VRAM, and driver against [NVIDIA's Isaac Sim 6.0 system requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html#system-requirements).
2. Install the [Isaac Sim 6.0 workstation distribution](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_workstation.html). The restaurant experiment uses Windows, DirectX 12, and Python 3.12.
3. From the repository root, set up Python and the perception dependencies.

   ```powershell
   conda create -n cra python=3.12 -y
   conda activate cra
   python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
   python -m pip install -r experiment/requirements.txt
   ```

   Get the [model weights](experiment/models/README.md) and follow the [experiment guide](experiment/README.md) to prepare the restaurant scene.

## Citation

[Read the paper on arXiv](https://arxiv.org/abs/2611.22080).

```bibtex
@misc{zheng2026cracontinuousriskassessment,
  title={CRA: Continuous Risk Assessment for Safety-critical Visual Navigation},
  author={Lihao Zheng and Ying Wu and Chen Jiang and Randy Freeman},
  year={2026},
  eprint={2611.22080},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2611.22080}
}
```
