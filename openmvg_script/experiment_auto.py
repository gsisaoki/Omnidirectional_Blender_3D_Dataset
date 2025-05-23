import os
import numpy as np

def remove_points_from_ply(input_ply_path, output_ply_path, remove_first_n=100, remove_last_n=0):
    with open(input_ply_path, 'r') as f:
        lines = f.readlines()

    end_header_idx = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
    
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_line_idx = i
            vertex_count = int(line.split()[-1])
            break

    points = lines[end_header_idx + 1 : end_header_idx + 1 + vertex_count]

    start = remove_first_n
    end = vertex_count - remove_last_n
    filtered_points = points[start:end]

    new_vertex_count = len(filtered_points)
    lines[vertex_line_idx] = f"element vertex {new_vertex_count}\n"

    with open(output_ply_path, 'w') as f:
        f.writelines(lines[:end_header_idx + 1])
        f.writelines(filtered_points)

database_path = "/your/path/to/dataset"

scene_names = os.listdir(database_path)

scene_names = ["pavillion"]

output_base_path = "/your/path/to/output"

plys_path = os.path.join(output_base_path, "plys")

for scene_name in scene_names:
    scene_path = os.path.join(database_path, scene_name)

    for i in ["Egocentric", "Non-Egocentric"]:
        images_path = os.path.join(scene_path, i)
        images_path = os.path.join(images_path, "images")
        output_path = os.path.join(output_base_path, scene_name)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = os.path.join(output_path, i)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            continue

        matches_path = os.path.join(output_path, "matches")
        recon_path = os.path.join(output_path, "reconstruction")

        if not os.path.exists(matches_path):
            os.makedirs(matches_path)

        if not os.path.exists(recon_path):
            os.makedirs(recon_path)

        os.system(f"openMVG_main_SfMInit_ImageListing -i {images_path} -o {matches_path} -f 1 -c 7")
        os.system(f"openMVG_main_ComputeFeatures -i {matches_path}/sfm_data.json -o {matches_path} -p ULTRA") # Set -p to “ULTRA” to increase the number of points
        os.system(f"openMVG_main_PairGenerator -i {matches_path}/sfm_data.json -o {matches_path}/pairs.bin")
        os.system(f"openMVG_main_ComputeMatches -i {matches_path}/sfm_data.json -p {matches_path}/pairs.bin -o {matches_path}/matches.putative.bin -r 0.3") # Set -r to “0.3” to increase the number of points
        os.system(f"openMVG_main_GeometricFilter -i {matches_path}/sfm_data.json -m {matches_path}/matches.putative.bin -g a -o {matches_path}/matches.f.bin")
        os.system(f"openMVG_main_SfM -i {matches_path}/sfm_data.json -m {matches_path} -o {recon_path}")
        os.system(f"openMVG_main_ConvertSfM_DataFormat -i {recon_path}/sfm_data.bin -o {recon_path}/sfm_data.json")
        os.system(f"openMVG_main_ConvertSfM_DataFormat -i {recon_path}/sfm_data.bin -o {recon_path}/{scene_name}.ply -S")
        os.system(f"openMVG_main_ComputeSfM_DataColor -i {recon_path}/sfm_data.bin -o {recon_path}/{scene_name}_color.ply")
        # Remove camera points
        remove_points_from_ply(
            input_ply_path=os.path.join(recon_path, f"{scene_name}_color.ply"),
            output_ply_path=os.path.join(plys_path, f"{scene_name}_{i}.ply"),
            remove_first_n=0,
            remove_last_n=100
        )
    