import numpy as np
import OpenEXR
import Imath
import torch

def read_exr_normal(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    normal_x = np.frombuffer(exr_file.channel('X', pt), dtype=np.float32).reshape((height, width))
    normal_y = np.frombuffer(exr_file.channel('Y', pt), dtype=np.float32).reshape((height, width))
    normal_z = np.frombuffer(exr_file.channel('Z', pt), dtype=np.float32).reshape((height, width))
    normal = np.stack([normal_x, normal_y, normal_z], axis=-1)

    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm[norm == 0] = 1e-8
    normal = normal / norm
    normal = np.clip(normal, -1.0, 1.0)
    normal = (normal + 1) / 2 
    normal = np.clip(normal, 0, 1)
    normal = np.flip(normal, axis=2)
    return normal

# from DSINE (https://github.com/baegwangbin/DSINE/blob/main/utils/utils.py)
def compute_normal_error(pred_norm, gt_norm):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    return pred_error

def compute_normal_metrics(total_normal_errors):
    """ compute surface normal metrics (used for benchmarking)
        NOTE: total_normal_errors should be a 1D torch tensor of errors in degrees
    """
    total_normal_errors = total_normal_errors.detach().cpu().numpy()
    num_pixels = total_normal_errors.shape[0]

    metrics = {
        'mean':   np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse':   np.sqrt(np.sum(total_normal_errors * total_normal_errors) / num_pixels),
        'a5':     100.0 * (np.sum(total_normal_errors < 5) / num_pixels),
        'a7.5':   100.0 * (np.sum(total_normal_errors < 7.5) / num_pixels),
        'a11.25': 100.0 * (np.sum(total_normal_errors < 11.25) / num_pixels),
        'a22.5':  100.0 * (np.sum(total_normal_errors < 22.5) / num_pixels),
        'a30':    100.0 * (np.sum(total_normal_errors < 30) / num_pixels)
    }
    return metrics

def normalize_vector_map(v):
    norm = torch.norm(v, dim=1, keepdim=True)
    norm = torch.where(norm == 0, torch.tensor(1e-6), norm)
    return v / norm

def compute_normal_metrics_from_numpy(pred_norm, gt_norm):
    normal_map_reshaped = torch.tensor(pred_norm.copy()).permute(2, 0, 1).unsqueeze(0)
    normal_map_gt_reshaped = torch.tensor(gt_norm.copy()).permute(2, 0, 1).unsqueeze(0)
    normal_map_reshaped = normalize_vector_map(normal_map_reshaped)
    normal_map_gt_reshaped = normalize_vector_map(normal_map_gt_reshaped)
    normal_error = compute_normal_error(normal_map_reshaped, normal_map_gt_reshaped)
    normal_error_flat = normal_error.view(-1)
    normal_metrics = compute_normal_metrics(normal_error_flat)
    return normal_metrics