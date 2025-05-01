import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import os
import random
from pytorch3d.transforms import so3_relative_angle
from itertools import combinations


def camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.

    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """

    with torch.no_grad():
        # Convert cameras to 4x4 SE3 transformation matrices
        gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()
        pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

        # Generate pairwise indices to compute relative poses
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, gt_se3.shape[0] // batch_size)
        pair_idx_i1 = pair_idx_i1.to(device)

        # Compute relative camera poses between pairs
        # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
        # This is possible because of SE3
        relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
        relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

        # Compute the difference in rotation and translation
        # between the ground truth and predicted relative camera poses
        rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
        rel_tangle_deg = translation_angle(relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3])

    return rel_rangle_deg, rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))


def calculate_auc(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Define histogram bins
    bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()


def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


def closed_form_inverse(se3):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    """
    R = se3[:, :3, :3]
    T = se3[:, 3:, :3]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # Compute the left part of the inverse transformation
    left_bottom = -T.bmm(R_transposed)
    left_combined = torch.cat((R_transposed, left_bottom), dim=1)

    # Keep the right-most column as it is
    right_col = se3[:, :, 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-4)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-20, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def compute_ARE(rotation1, rotation2):
    if isinstance(rotation1, torch.Tensor):
        rotation1 = rotation1.cpu().detach().numpy()
    if isinstance(rotation2, torch.Tensor):
        rotation2 = rotation2.cpu().detach().numpy()

    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    error = theta * 180 / np.pi
    return np.minimum(error, np.abs(180 - error))

def load_camera_params(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)

        data = data[0]

        rotation = np.array(data['extrinsics']['rotation']) # c2w
        translation = np.array(data['extrinsics']['translation']) # c2w
    
    return [rotation, translation]

def load_camera_gt(folder_path):
    cameras = []
    files = sorted(os.listdir(folder_path))
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = load_camera_params(file_path)
        cameras.append(data)
    return cameras

def load_camera_openmvg(json_path):
    with open(json_path, 'r') as f:
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
        t_w2c = -R_c2w @ t_c2w

        camera_params_list.append([R_c2w, t_c2w, filename, pose_id])

    poses = [[np.array(R), np.array(t)] for R, t, _, _ in camera_params_list]

    return poses
    
def pose_list_to_se3_tensor(pose_list):
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

def evaluate_camera(gt_cameras_path, pred_cameras_path, output_path=None):

    gt_cameras = load_camera_gt(gt_cameras_path)
    pred_cameras = load_camera_openmvg(pred_cameras_path)

    gt_se3 = pose_list_to_se3_tensor(gt_cameras)
    pred_se3 = pose_list_to_se3_tensor(pred_cameras)

    relative_pose_gt, relative_pose_pred = make_relative_pose(gt_se3, pred_se3)

    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    print("Relative rotation angle (degrees) :", rel_rangle_deg.mean().item())
    print("Relative translation angle (degrees) :", rel_tangle_deg.mean().item())

    auc_30 = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=30)
    print("AUC (30 degrees):", auc_30)
    auc_10 = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=10)
    print("AUC (10 degrees):", auc_10)
    auc_5 = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=5)
    print("AUC (5 degrees):", auc_5)

    ate = compute_ate(gt_se3[:, :3, 3].cpu().numpy(),pred_se3[:, :3, 3].cpu().numpy())
    print("ATE (adjacent pairs):", ate)

    if output_path is not None:
        pass