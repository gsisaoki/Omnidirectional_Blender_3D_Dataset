import os
import sys
import json
import numpy as np

args = sys.argv
sfm_data_path = args[1] # /path/to/sfm_data.json
extrinsics_dir = args[2] # /path/to/scene/capture_type/cameras
save_dir = os.path.join(args[3], 'matches') # /path/to/output

with open(sfm_data_path, 'r') as sfm_data:
    sfm_data = json.load(sfm_data)

    if os.path.isdir(extrinsics_dir):
        extrinsics_files = sorted(os.listdir(extrinsics_dir))
        
        for i, extrinsic_file in enumerate(extrinsics_files):
            
            if extrinsic_file.endswith('.json'):
                extrinsic_file = os.path.join(extrinsics_dir, extrinsic_file)
                
                ### Get gt camera pose
                with open(extrinsic_file, 'r') as f:
                    extrinsic_data = json.load(f)[0]['extrinsics']
                    translation = np.array(extrinsic_data['translation'])
                    rotation = np.array(extrinsic_data['rotation'])

                    # P = np.eye(4)
                    # P[:3, :3] = rotation
                    # P[:3, 3] = translation
                    # P_inv = np.linalg.inv(P)
                    # rotation_c2w = P_inv[:3, :3]
                    # translation_c2w = P_inv[:3, 3]

                    # center = -rotation_c2w.T @ translation_c2w
                    
                    center = -rotation.T @ translation
                    
                    print(center)
                    key = i
                    sfm_data["extrinsics"].append({"key": key, "value": {"rotation": rotation.tolist(), "center": center.tolist()}})
            
            else:
                print("Not a json file: ", extrinsic_file)
                exit()
    
    else:
        print("Not a directory: ", extrinsics_dir)
        exit()
        
    new_sfm_data = sfm_data

with open(os.path.join(save_dir, "sfm_data_gtpose.json"), 'w') as f:
    json.dump(new_sfm_data, f, indent=4)
    
print("\n---------------------------")
print("Format conversion has DONE!")
print("---------------------------\n")