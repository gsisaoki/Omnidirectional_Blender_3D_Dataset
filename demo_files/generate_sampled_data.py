import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sampled data for NeurIPS.')
    parser.add_argument('--base_dir', type=str, default='/path/to/OB3D/')
    parser.add_argument('--num_interval', type=int, default=4)
    parser.add_argument('--scene', type=str, default='EmeraldSquare')
    parser.add_argument('--output_dir', type=str, default='/path/to/output/')
    args = parser.parse_args()

    base_dir = args.base_dir
    num_interval = args.num_interval
    scene = args.scene
    output_dir = args.output_dir

    capture_types = ['Egocentric', 'Non-Egocentric']
    for capture_type in capture_types:
        data_path = os.path.join(base_dir, scene, capture_type)
        depth_path_list = sorted(glob.glob(os.path.join(data_path, 'depths/*.exr')))
        normal_path_list = sorted(glob.glob(os.path.join(data_path, 'normals/*.exr')))
        image_path_list = sorted(glob.glob(os.path.join(data_path, 'images/*.png')))
        camera_path_list = sorted(glob.glob(os.path.join(data_path, 'cameras/*.json')))
        
        output_path = os.path.join(output_dir, scene + '_sampled', capture_type)
        os.makedirs(os.path.join(output_path, 'depths'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'cameras'), exist_ok=True)

        for i in range(0, len(depth_path_list), num_interval):
            depth_path = depth_path_list[i]
            normal_path = normal_path_list[i]
            image_path = image_path_list[i]
            camera_path = camera_path_list[i]

            shutil.copy(depth_path, os.path.join(output_path, 'depths', os.path.basename(depth_path)))
            shutil.copy(normal_path, os.path.join(output_path, 'normals', os.path.basename(normal_path)))
            shutil.copy(image_path, os.path.join(output_path, 'images', os.path.basename(image_path)))
            shutil.copy(camera_path, os.path.join(output_path, 'cameras', os.path.basename(camera_path)))

        print(f"Sampled data copied to {output_path}")