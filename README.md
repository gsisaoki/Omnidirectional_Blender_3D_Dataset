# OmniBlender3D

## Overview
This repository provides some codes for [Omnidirectional Blender 3D Dataset (OB3D)](https://www.kaggle.com/datasets/shintacs/ob3d-dataset), a dataset designed for 3D reconstruction from multi-view equirectangular images.
In addition to 3D reconstruction, OB3D also supports novel view synthesis and camera pose estimation for equirectangular images.
This dataset consists of 12 scenes, each of which contains RGB images, depth maps, normal maps, camera parameters, and sparse 3D point clouds.


## Dataset Structure
The dataset is organized as follows:

```
OB3D
|-- archivis-flat
|   |-- Non-Egocentric
|       |--cameras
|          |--00000_cam.json
|          |--...
|       |--images
|          |--00000_rgb.png
|          |--...
|       |--depths
|          |--00000_depth.exr
|          |--...
|       |--normals
|          |--00000_normal.exr
|          |--...
|   |-- Egocentric
|       |-...
```
OB3D can be downloaded from [OB3D](https://www.kaggle.com/datasets/shintacs/ob3d-dataset) where more detailed information about the OB3D is also provided.

## How to evaluate a reconstructed mesh
To evaluate a reconstructed mesh using OB3D, it is required to reconstruct the mesh in the same coordinate system and scale as the ground truth, using the provided ground-truth camera parameters.

Once the mesh is reconstructed in the same scale and coordinate system as the ground truth, our evaluation code can be used to quantitatively evaluate its qualit. 

A minimal example of how to run the evaluation is shown below:

```python
import numpy as np
from utils.equirectangular_render import *
from utils.eval_depth import *

# Step 1: Rendring depth map using reconstructed mesh and g.t. camera parameter
depth_map, normal_map = equirectangular_renderer_from_mesh(ply_path, camera_path, width=1600, height=800)

# Step 2: Quantitative evaluation 
depth_map_gt = np.array(read_exr_depth(gt_depth_path)).astype(np.float32)
depth_metrics = calculate_metrics(depth_map, depth_map_gt, depth_max_value=20.0)
print(depth_metrics)
```
- ply_path: path to reconstructed mesh
- camera_path: path to ground truth camera parameter (e.g. 00000_cam.json)
- gt_depth_path: path to ground truth depth map (e.g. 00000_depth.exr)
## Example of 3D Reconstruction (only support qualitative evaluation)
<details>
<summary>Details</summary>
We show an example usage of our dataset uging OmniSDF(CVPR2024) (support only qualitatitive evaluation)

1. Download code 
    ```
    git clone https://github.com/KAIST-VCLAB/OmniSDF.git
    cd OmniSDF
    ```
2. Preparetion
    - Make the necessary modifications to the dataloader of your mehod so that it can load our dataset.
    - In this case, we provide a modified version of the OmniSDF dataloader that supports our dataset. The dataloader is in ./demo_files/dataset_omniphoto.py
3. Make Config file
    <details> 
    <summary>./confs/demo.conf</summary>

    ```
    general {
	    base_exp_dir = /path/to/output
        recording = [
        ./,
        ./models
    ]
    debug = False
    summary_image = True
    dataset_classname = Blender360
    is_continue = -1
    }
    dataset {
    data_dir = /path/to/OmniBlender3D/scene/
    fr_start = 0
    fr_end = 100     # Total number of input images
    fr_interval = 1
    fr_scale = 1.0
    world_scale = 1.0
    far_sphere_bound = 30   # Radius enclosing the scene
    obj_bbox_max = [1.01, 1.01, 1.01]
    obj_bbox_min = [-1.01, -1.01, -1.01]
    dataset_name = Blender360
    }
    ...
    ```
    </details>
4. Train
    ```
    CUDA_VISIBLE_DEVICES=1 python main.py --mode=train --conf="./confs/demo.conf"
    ```
5. Extract Mesh
    ```
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --mode=validate_mesh \
        --conf="./confs/demo.conf" \
        --is_continue
    ```
</details>

## Additional Information
### 1. Change the number of input views
If you wish to adjust the number of viewpoints, you can easily create a modified version of the dataset by running the following command:
```
python demo_files/generate_sampled_data.py \
    --base_fir /path/to/OmniBlender3D \
    --scene archiviz-flat \
    --output_dir /path/to/modified-dataset
```
### 2. Evaluate a mesh reconstructed SDF-based method
To evaluate a reconstructed mesh using our dataset, it is essential to reconstruct the mesh in the same coordinate system and scale as the ground truth, using the provided ground-truth camera parameters. 
However, in some methods like SDF-based methods, it may be necessary to transform the scene into a normalized space—such as fitting it into a unit sphere—which alters the scale and coordinate system. 
In such cases, we recommend saving the transformation parameters so that the mesh can be converted back to the original coordinate system and scale for evaluation.
