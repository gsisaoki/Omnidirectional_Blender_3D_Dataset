import numpy as np
import Imath
import OpenEXR

def read_exr_depth_v2(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channel = 'V'
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel(channel, pt)
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = np.reshape(depth, (height, width))
    return depth

def calculate_mae(est_depth, gt_depth, valid_mask, num_valid_pixels):
    mae = np.sum(np.abs(gt_depth[valid_mask] - est_depth[valid_mask])) / num_valid_pixels
    return mae

def calculate_mse(est_depth, gt_depth, valid_mask, num_valid_pixels): 
    mse = np.sum(np.square(gt_depth[valid_mask] - est_depth[valid_mask])) / num_valid_pixels
    return mse

def calculate_rmse(est_depth, gt_depth, valid_mask, num_valid_pixels):
    rmse = np.sqrt(calculate_mse(est_depth, gt_depth, valid_mask, num_valid_pixels))
    return rmse

def calculate_rmse_log(est_depth, gt_depth, valid_mask, num_valid_pixels):
    rmse_log = np.sqrt(np.sum(np.square(np.log(gt_depth[valid_mask]) - np.log(est_depth[valid_mask]))) / num_valid_pixels)
    return rmse_log

def calculate_rmse_scale_invariant(est_depth, gt_depth, valid_mask, num_valid_pixels):
    log_diff = np.log(est_depth[valid_mask]) - np.log(gt_depth[valid_mask])
    rmse_scale_invariant  = np.sqrt(np.sum(np.square(log_diff)) / num_valid_pixels - np.square(np.sum(log_diff)) / np.square(num_valid_pixels))
    return rmse_scale_invariant

def calculate_abs_relative_difference(est_depth, gt_depth, valid_mask, num_valid_pixels):
    abs_relative_difference = np.sum(np.abs(gt_depth[valid_mask] - est_depth[valid_mask]) / gt_depth[valid_mask]) / num_valid_pixels
    return abs_relative_difference

def calculate_squared_relative_difference(est_depth, gt_depth, valid_mask, num_valid_pixels):
    squared_relative_difference = np.sum(np.square(gt_depth[valid_mask] - est_depth[valid_mask]) / gt_depth[valid_mask]) / num_valid_pixels
    return squared_relative_difference

def calculate_percentage_within_threshold(est_depth, gt_depth, valid_mask, num_valid_pixels, ratio_threshold=1.25):
    ratio = np.maximum(est_depth[valid_mask] / gt_depth[valid_mask], gt_depth[valid_mask] / est_depth[valid_mask])
    percentage_within_threshold = np.sum(ratio < ratio_threshold) / num_valid_pixels
    return percentage_within_threshold

def calculate_metrics(est_depth, gt_depth, depth_max_value=None):
    if depth_max_value is not None:
        est_depth[est_depth > depth_max_value] = np.nan
        gt_depth[gt_depth > depth_max_value] = np.nan
        
    valid_mask = ~np.isnan(est_depth) & ~np.isnan(gt_depth)
    num_valid_pixels = np.sum(valid_mask)

    if num_valid_pixels == 0:
        raise ValueError("No valid pixels found in the depth maps.")

    mae = calculate_mae(est_depth, gt_depth, valid_mask, num_valid_pixels)
    mse = calculate_mse(est_depth, gt_depth, valid_mask, num_valid_pixels)
    rmse = calculate_rmse(est_depth, gt_depth, valid_mask, num_valid_pixels)
    rmse_log = calculate_rmse_log(est_depth, gt_depth, valid_mask, num_valid_pixels)
    rmse_scale_invariant = calculate_rmse_scale_invariant(est_depth, gt_depth, valid_mask, num_valid_pixels)
    abs_relative_difference = calculate_abs_relative_difference(est_depth, gt_depth, valid_mask, num_valid_pixels)
    squared_relative_difference = calculate_squared_relative_difference(est_depth, gt_depth, valid_mask, num_valid_pixels)
    percentage_within_threshold_1_25 = calculate_percentage_within_threshold(est_depth, gt_depth, valid_mask, num_valid_pixels)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "RMSE_scale_invariant": rmse_scale_invariant,
        "Percentage_within_threshold_1.25": percentage_within_threshold_1_25,
        "MSE": mse,
        "RMSE_log": rmse_log,
        "Abs_relative_difference": abs_relative_difference,
        "Squared_relative_difference": squared_relative_difference,
    }

    return metrics