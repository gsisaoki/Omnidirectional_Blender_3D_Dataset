# Omnidirectional Blender 3D (OB3D) Dataset

## Overview
This repository provides a set of codes for evaluating methods dedicated to tasks of equirectangular images using [Omnidirectional Blender 3D Dataset (OB3D)](https://www.kaggle.com/datasets/shintacs/ob3d-dataset).
OB3D is designed for evaluating the accuracy of 3D reconstruction from multi-view equirectangular images and can also be used to evaluate the accuracy of novel view synthesis and camera pose estimation.
OB3D consists of 12 scenes, each of which consists of RGB images, depth maps, normal maps, camera parameters, and sparse 3D point clouds.

## Dataset Structure of OB3D
The dataset structure of OB3D is shown below.

```
OB3D
|-- archivis-flat
|   |-- Egocentric
|       |--cameras
|          |--00000_cam.json
|          |--...
|       |--depths
|          |--00000_depth.exr
|          |--...
|       |--images
|          |--00000_rgb.png
|          |--...
|       |--normals
|          |--00000_normal.exr
|          |--...
|       |--sparse
|          |--sparse.ply
|       |--train.txt
|       |--test.txt
|   |-- Non-Egocentric
|       |-...
```

The data of OB3D is available at [OB3D](https://www.kaggle.com/datasets/shintacs/ob3d-dataset), where the detailed information is also provided.

## Evaluate a mesh model reconstructed by your method

Our evaluation code assumes that the reconstructed mesh model has the same coordinate system and scale as the ground truth.
If you evaluate the qulaity of recostructed mesh model by our code, you have to use the provied ground-truth camera parameters to reconstruct a mesh model from the equirectangulr images in OB3D by your method.

A minimal example of how to run the evaluation code is shown below, and a practical example can be found in `eval_demo.ipynb`.

```python
import numpy as np
from utils.equirectangular_render import *
from utils.eval_depth import *

# Step 1: Depth map rendering using the reconstructed mesh and the ground-truth camera parameters
depth_map, normal_map = equirectangular_renderer_from_mesh(ply_path, camera_path, width=1600, height=800)

# Step 2: Quantitative evaluation 
depth_map_gt = np.array(read_exr_depth(gt_depth_path)).astype(np.float32)
depth_metrics = calculate_metrics(depth_map, depth_map_gt, depth_max_value=20.0)
print(depth_metrics)
```
- ply_path: Path to reconstructed mesh
- camera_path: Path to the ground-truth camera parameter (e.g., 00000_cam.json)
- gt_depth_path: Path to ground truth depth map (e.g., 00000_depth.exr)

## Example of 3D Reconsturuction with NeuS (support quantitative and qualitative evaluation)
<details>
<summary>Details</summary>
We provide an example usage of our dataset uging NeuS (NeurIPS 2021).
Since NeuS is typically designed for perspective images, we provide a modified version of NeuS, called <a href=https://github.com/ShntrIto/SDF360/tree/main/confs>SDF360</a> which enables rendering of ERP images by modifying the ray generation method. 

1. Download codes
    ```
    git clone https://github.com/ShntrIto/SDF360.git
    cd SDF360
    ```
2. Preparetion
    - To train NeuS, it is necessary to preprocess the dataset according to the instructions provided in the [Training NeuS Using Your Custom Data](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data)
    - For a demonstration of mesh evaluation, we provide preprocessed data at <a href=https://github.com/ShntrIto/SDF360/tree/main/confs>SDF360</a>.
3. Make a config file
    <details> 
    <summary>./confs/womask_erp.conf</summary>

    ```
    general {
        base_exp_dir = ./exp/CASE_NAME
        recording = [
            ./,
            ./models
        ]
    }

    dataset {
        data_dir = ./dataset/CASE_NAME
        render_cameras_name = cameras_sphere.npz
        object_cameras_name = cameras_sphere.npz
        is_erp_image = True
        is_masked = True
    }
    train {
        learning_rate = 5e-4
        learning_rate_alpha = 0.05
        end_iter = 200000
    ...
    }
    ...(following setting is the same as NeuS)
    ```
    </details>
4. Train
    ```
    python main.py --mode train --conf ./confs/demo.conf
    ```
5. Extract a mesh model
    ```
    python main.py --mode validate_mesh --conf ./confs/demo.conf --is_continue
    ```
6. Evaluate a mesh model following the above description
</details>

## Example of 3D Reconstruction with OmniSDF(only support qualitative evaluation)
<details>
<summary>Details</summary>
We provide an example usage of our dataset uging OmniSDF (CVPR2024)

1. Download codes
    ```
    git clone https://github.com/KAIST-VCLAB/OmniSDF.git
    cd OmniSDF
    ```
2. Preparetion
    - Make the necessary modifications to the dataloader of your mehod so as to load our dataset.
    - We provide a modified version of the OmniSDF dataloader that supports our dataset. The dataloader is available at `./demo_files/dataset_omniphoto.py`
3. Make a config file
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
    data_dir = /path/to/OB3D/scene/
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
    python main.py --mode=train --conf="./confs/demo.conf"
    ```
5. Extract a mesh model
    ```
    python main.py \
        --mode=validate_mesh \
        --conf="./confs/demo.conf" \
        --is_continue
    ```
</details>

## Additional Information
### 1. Change the number of input views
If you would like to chenge the number of input views, you can easily create a modified version of the dataset by running the following command:
```
python demo_files/generate_sampled_data.py \
    --base_fir /path/to/OB3D \
    --scene archiviz-flat \
    --output_dir /path/to/modified-dataset
```
### 2. Evaluate a mesh model reconstructed by SDF-based methods
As mentiond above, it is essential to reconstruct the mesh model in the same coordinate system and scale as the ground truth. 
In some methods such as SDF-based methods, it may be necessary to transform the scene into a normalized space, such as fitting it into a unit sphere, which alters the scale and coordinate system. 
We recommend saving the transformation parameters so that the mesh model can be converted back to the original coordinate system and scale for evaluation.

## Evaluate a RGB image rendered by Novel View Synthesis (NVS) method

### 1. Set the path to rendered and GT images

### 2. Run the command below
- Evaluate rendered image in terms of PSNR[dB], SSIM, LPIPS(A) and LPIPS(V).

```python

device = torch.device("cuda:0")
torch.cuda.set_device(device)

gt_path = "/path/to/your/ground/truth/image"
render_path = "/path/to/your/rendered/image"
render, gt = read_image(render_path, gt_path)
metrics = calculate_metrics(render, gt)

print("---------------------------------")
print("  PSNR        : {:>12.7f}".format(metrics['PSNR'], ".5"))
print("  SSIM        : {:>12.7f}".format(metrics['SSIM'], ".5"))
print("  LPIPS(alex) : {:>12.7f}".format(metrics['LPIPS(alex)'], ".5"))
print("  LPIPS(vgg)  : {:>12.7f}".format(metrics['LPIPS(vgg)'], ".5"))
print("---------------------------------")

```

## Evaluate estimated camera poses

### 1. Estimate camera poses and save them in the following format of OpenSfM or OpenMVG

### 2. Run the command belos

```python
from utils.eval_camera import *
 
gt_camera_path = "/path/to/OB3D/scene/Egocentric/cameras"
 
# This dirctory assume '/path/to/estimated/camera/openmvg/reconstruction/sdf_data.json'
openmvg_camera_path = "/path/to/estimated/camera/openmvg"       
camera_metrics_openmvg = evaluate_camera(gt_camera_path, openmvg_camera_path, pred_cameras_type="openmvg")
 
# This dirctory assume '/path/to/estimated/camera/opensfm/reconstruction.json'
opensfm_camera_path = "/path/to/estimated/camera/opensfm"       
camera_metrics_opensfm = evaluate_camera(gt_camera_path, opensfm_camera_path, pred_cameras_type="opensfm")
```

## Acknowledgement
- In our demonstration of OB3D, we used [OmniSDF](https://github.com/KAIST-VCLAB/OmniSDF) and [SDF360](https://github.com/ShntrIto/SDF360/tree/main) (NeuS modified for ERP images).
The code of SDF360 is built upon [NeuS](https://github.com/Totoro97/NeuS).
- We refered to the implementation of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) and [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim%5C) for creating the evaluation code for novel view synthesis.
Thank you for all of these great projects.
- For camera parameter estimation, we used two methods, [OpenSfM](https://opensfm.org/) and [OpenMVG](https://github.com/openMVG/openMVG).
- Thank you for all of these great projects.