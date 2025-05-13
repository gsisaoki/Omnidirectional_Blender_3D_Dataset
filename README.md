# OmniBlender3D

## Overview
This repository provides the **OmniBlender360**, a dataset designed for 3D reconstruction from multi-view equirectangular images.
This dataset consists of XXX scenes, each of which contains RGB images, depth maps, normal maps, and camera parameters.


## Dataset Structure
The dataset is organized as follows:

```
OmniBlender3D
|-- archivis-flat
|   |-- Non-Egocentric
|       |--cameras
|          |--00000_cam.json
|          |--...
|       |--depths
|          |--00000_rgb.png
|          |--...
|       |--images
|          |--00000_depth0001.exr
|          |--...
|       |--normals
|          |--00000_normal0001.exr
|          |--...
```

## Example Usage 
We show an example usage of our dataset uging OmniSDF(CVPR2024)

1. Download code 
    <pre><code>
    git clone https://github.com/KAIST-VCLAB/OmniSDF.git
    cd OmniSDF
    </code></pre>
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
    <pre><code>
    CUDA_VISIBLE_DEVICES=1 python main.py --mode=train --conf="./confs/demo.conf"
    </code></pre>
5. Extract Mesh
    <pre><code>
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --mode=validate_mesh \
        --conf="./confs/demo.conf" \
        --is_continue
    </code></pre>
6. Additional Information
    - If you wish to adjust the number of viewpoints, you can easily create a modified version of the dataset by running the following command:
    <pre><code>
    python demo_files/generate_sampled_data.py \
        --base_fir /path/to/OmniBlender3D \
        --scene archiviz-flat \
        --output_dir /path/to/modified-dataset

    </code></pre>
