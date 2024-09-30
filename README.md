# Double Decomposition for Nighttime Driving Scene Simulation


## Introduction

This is the official repository for *Double Decomposition for Nighttime Driving Scene Simulation*. In this repository, we release the Waymo-Night and nuScenes-Night dataset, as well as the code.

In this work, we propose a double decomposition method for nighttime driving scene simulation.
Our approach is centered around a double decomposition strategy, which divides the simulation process into two key components: intrinsic and static-dynamic decomposition. 
<!-- ![Nighttime Driving Scene](README.assets/teaser.png) -->

## Installation

1. Create conda environment:
```Bash
  conda create -n nighttime-stgs python=3.8
  conda activate nighttime-stgs

  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

  # Install requirements
  pip install -r requirments.txt

  # Install submodules
  pip install ./submodules/diff-gaussian-rasterization
  pip install ./submodules/simple-knn
  pip install ./submodules/simple-waymo-open-dataset-reader
  python script/test_gaussian_rasterization.py
  pip install -r requirements.txt
```

## Dataset

2. Prepare for dataset:
   We use nighttime waymo dataset following [EmerNeRF](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md)

```
data
|__Waymo-Night
   |__Package name (e.g. 007)
      |__dynamic_mask
      |__ego_pose
      |__extrinsics
      |__gt_depth
      |__images
      |__intrinsics
      |__lidar_depth
      |__sky_mask
   nuScenes-Night
   |__sequences (e.g. scene-1100)
      |__aggregate_lidar
      |__colmap
      |__depths
      |__depths_lidar_patch5_new
      |__images
      |__egomasks
      |__lidars
      |__masks
      |__segs
```

## Train
```Bash
   bash script/waymo/train_waymo_exp.sh
```

## Render

```Bash
   bash script/waymo/render_waymo_exp.sh
```

<!-- 1. Test on Waymo-Night 
```shell
![image-20211002220051137](README.assets/teaser.png)
```

2. Test on nuScenes-Night
```shell
![image-20211002220051137](README.assets/teaser.png)
``` -->

## Citation
If you find this work useful for your research, please cite our paper:
```shell
   todo
```

## Acknowledgement
We would like to thank the reviewers for their constructive comments and the authors of [SCI](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.html) and [StreetGaussians](https://github.com/zju3dv/street_gaussians) for their help and suggestions.