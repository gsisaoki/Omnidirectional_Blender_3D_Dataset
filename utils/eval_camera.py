import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import os
import cv2
from collections import defaultdict

import utils.camera_metric as metric

def nested_dict():
    return defaultdict(nested_dict)

def load_camera_params(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)
        data = data[0]
        
        rotation_w2c = np.array(data['extrinsics']['rotation']) # w2c
        translation_w2c = np.array(data['extrinsics']['translation']) # w2c
        rotation_c2w = rotation_w2c.T
        translation_c2w = -rotation_c2w @ translation_w2c
    
    return [rotation_w2c, translation_w2c]

def load_camera_gt(folder_path):
    cameras = []
    files = sorted(os.listdir(folder_path))
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = load_camera_params(file_path)
        cameras.append(data)
    return cameras

def load_camera_openmvg(folder_path):
    with open(os.path.join(folder_path, "reconstruction", "sfm_data.json"), 'r') as f:
        data = json.load(f)

    view_info = []
    for view in data['views']:
        v = view['value']['ptr_wrapper']['data']
        filename = v['filename']
        pose_id = v['id_pose']
        view_info.append((filename, pose_id))

    view_info.sort()

    extrinsics = {e['key']: e['value'] for e in data['extrinsics']}

    camera_params_list = []

    for filename, pose_id in view_info:
        extrinsic = extrinsics.get(pose_id)

        R_w2c = np.array([[float(x) for x in row] for row in extrinsic['rotation']])
        R_c2w = np.linalg.inv(R_w2c)         
        t_c2w = np.array(extrinsic['center']) 
        t_w2c = -R_w2c @ t_c2w

        camera_params_list.append([R_w2c, t_w2c, filename, pose_id])

    poses = [[np.array(R), np.array(t)] for R, t, _, _ in camera_params_list]

    return poses

def load_camera_opensfm(folder_path):
    with open(os.path.join(folder_path, "reconstruction.json"), "r") as f:
            data = json.load(f)
            data = data[0]
            shots = data["shots"]

    cameras = []
    
    sorted_shots = sorted(shots.items(), key=lambda x: x[0])

    for name, shot in sorted_shots:
        
        rotation_vector = np.array(shot["rotation"], dtype=np.float64)  # w2c
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        translation = np.array(shot["translation"], dtype=np.float64)  # w2c

        cameras.append([rotation_matrix, translation])
    
    return cameras

    
def pose_list_to_se3_temsor(pose_list):
    se3_matrices = []

    for R, t, *_ in pose_list:
        se3 = np.eye(4)
        se3[:3, :3] = R
        se3[:3, 3] = t
        se3_matrices.append(se3)

    se3_tensor = torch.tensor(np.stack(se3_matrices), dtype=torch.float32) # (N, 4, 4)

    return se3_tensor

def points_alignment(X, Y, with_scale=True):

    assert X.shape == Y.shape
    n = X.shape[0]
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)

    Xc = X - mean_X
    Yc = Y - mean_Y

    var_X = np.sum(np.sum(Xc**2, axis=1)) / n

    # Correlation matrix
    cov = Xc.T @ Yc / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    if with_scale:
        s = np.trace(np.diag(D) @ S) / var_X
    else:
        s = 1.0

    t = mean_Y - s * R @ mean_X

    return s, R, t

def make_relative_pose(gt_se3, pred_se3):

    n = gt_se3.shape[0]
    pairs = list(combinations(range(n), 2))
    pair_idx_i1 = torch.tensor([i for i, j in pairs], dtype=torch.long)
    pair_idx_i2 = torch.tensor([j for i, j in pairs], dtype=torch.long)

    gt_R_w2c = gt_se3[:, :3, :3]
    gt_t_w2c = gt_se3[:, :3, 3:4]
    gt_t_c2w = -np.matmul((np.transpose(gt_R_w2c, (0, 2, 1))), gt_t_w2c)
    gt_t_c2w = gt_t_c2w.squeeze(2)  # shape (N, 3)
    gt_se3[:, :3, 3] = gt_t_c2w

    pred_R_w2c = pred_se3[:, :3, :3]
    pred_t_w2c = pred_se3[:, :3, 3:4]
    pred_t_c2w = -np.matmul((np.transpose(pred_R_w2c, (0, 2, 1))), pred_t_w2c)
    pred_t_c2w = pred_t_c2w.squeeze(2)  # shape (N, 3)
    pred_se3[:, :3, 3] = pred_t_c2w

    s, R, t = points_alignment(pred_se3[:, :3, 3].numpy(), gt_se3[:, :3, 3].numpy(), with_scale=True)

    pred_se3[:, :3, 3] = s * pred_se3[:, :3, 3] @ R + t

    pair_idx_i1 = np.asarray(pair_idx_i1)
    pair_idx_i2 = np.asarray(pair_idx_i2)

    gt_i = gt_se3[pair_idx_i1]  # (M, 4, 4)
    gt_j = gt_se3[pair_idx_i2]  # (M, 4, 4)
    pred_i = pred_se3[pair_idx_i1]
    pred_j = pred_se3[pair_idx_i2]

    gt_Ri = gt_i[:, :3, :3]
    gt_Rj = gt_j[:, :3, :3]
    gt_ti = gt_i[:, :3, 3]
    gt_tj = gt_j[:, :3, 3]

    pred_Ri = pred_i[:, :3, :3]
    pred_Rj = pred_j[:, :3, :3]
    pred_ti = pred_i[:, :3, 3]
    pred_tj = pred_j[:, :3, 3]

    gt_R_rel = np.matmul(gt_Ri, np.transpose(gt_Rj, (0, 2, 1)))
    pred_R_rel = np.matmul(pred_Ri, np.transpose(pred_Rj, (0, 2, 1)))

    gt_t_rel = (gt_ti - gt_tj)[..., np.newaxis]  # shape (M, 3, 1)
    pred_t_rel = (pred_ti - pred_tj)[..., np.newaxis]

    relative_pose_gt = np.concatenate([gt_R_rel, gt_t_rel], axis=2)   # shape (M, 3, 4)
    relative_pose_pred = np.concatenate([pred_R_rel, pred_t_rel], axis=2)

    relative_pose_gt = torch.from_numpy(relative_pose_gt).float()
    relative_pose_pred = torch.from_numpy(relative_pose_pred).float()

    return relative_pose_gt, relative_pose_pred

# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse

def evaluate_camera(gt_cameras_path, pred_cameras_path, pred_cameras_type, output_path=None):

    gt_cameras = load_camera_gt(gt_cameras_path)
    print("GT cameras:", gt_cameras)

    if pred_cameras_type == "opensfm":
        pred_cameras = load_camera_opensfm(pred_cameras_path)
    elif pred_cameras_type == "openmvg":
        pred_cameras = load_camera_openmvg(pred_cameras_path)
    else:
        raise ValueError("Unknown camera type: {}".format(pred_cameras_type))
    print("Pred cameras:", pred_cameras)

    gt_se3 = pose_list_to_se3_temsor(gt_cameras)
    pred_se3 = pose_list_to_se3_temsor(pred_cameras)

    relative_pose_gt, relative_pose_pred = make_relative_pose(gt_se3, pred_se3)

    rel_rangle_deg = metric.rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = metric.translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    # print("Relative rotation angle (degrees) :", rel_rangle_deg.mean().item())
    # print("Relative translation angle (degrees) :", rel_tangle_deg.mean().item())

    auc_30 = metric.calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=30)
    # print("AUC (30 degrees):", auc_30)
    auc_10 = metric.calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=10)
    # print("AUC (10 degrees):", auc_10)
    auc_5 = metric.calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=5)
    # print("AUC (5 degrees):", auc_5)

    ate = compute_ate(gt_se3[:, :3, 3].cpu().numpy(),pred_se3[:, :3, 3].cpu().numpy())
    # print("ATE (adjacent pairs):", ate)

    if output_path is not None:
        pass

    return rel_rangle_deg, rel_tangle_deg, auc_30, auc_10, auc_5, ate