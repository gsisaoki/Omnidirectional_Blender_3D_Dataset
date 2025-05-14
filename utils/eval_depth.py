import numpy as np
import Imath
import OpenEXR
import json

def read_exr_depth(file_path, scale=1):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
    pixel_type = header['channels']['B'].type # 型判別用の変数
        
    if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
        # FLOAT を使う場合
        # print("The EXR file is stored in FLOAT format.")
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT) # archiviz-flat を使った実行用
        try:
            depth_str = exr_file.channel('V', FLOAT)
        except:
            try:
                depth_str = exr_file.channel('B', FLOAT)
            except:
                try:
                    depth_str = exr_file.channel('G', FLOAT)
                except:
                    try:
                        depth_str = exr_file.channel('R', FLOAT)
                    except:
                        raise ValueError("No valid depth channel found in the EXR file.")
        depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
    elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
        # HALF を使う場合
        # Read the depth channel as 16-bit floats
        # print("The EXR file is stored in HALF format.")
        HALF = Imath.PixelType(Imath.PixelType.HALF)
        depth_str = exr_file.channel('B', HALF)
        # Convert the binary string to a numpy array
        depth = np.frombuffer(depth_str, dtype=np.float16).reshape(size[1], size[0])
    else:
        print("The EXR file has an unsupported pixel type.")
    depth = depth*scale
    return depth

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

# TODO: avoid Nan
def calculate_rmse_log(est_depth, gt_depth, valid_mask, num_valid_pixels):
    # rmse_log = np.sqrt(np.sum(np.square(np.log(gt_depth[valid_mask]) - np.log(est_depth[valid_mask]))) / num_valid_pixels)
    rmse_log = np.sqrt(np.sum(np.square(np.log(gt_depth[valid_mask] + 1e-6) - np.log(est_depth[valid_mask] + 1e-6))) / num_valid_pixels)
    return rmse_log

# TODO: avoid Nan
def calculate_rmse_scale_invariant(est_depth, gt_depth, valid_mask, num_valid_pixels):
    # log_diff = np.log(est_depth[valid_mask]) - np.log(gt_depth[valid_mask])
    log_diff = np.log(gt_depth[valid_mask] + 1e-6) - np.log(est_depth[valid_mask] + 1e-6)
    rmse_scale_invariant  = np.sqrt(np.sum(np.square(log_diff)) / num_valid_pixels - np.square(np.sum(log_diff)) / np.square(num_valid_pixels))
    return rmse_scale_invariant

def calculate_abs_relative_difference(est_depth, gt_depth, valid_mask, num_valid_pixels):
    abs_relative_difference = np.sum(np.abs(gt_depth[valid_mask] - est_depth[valid_mask]) / gt_depth[valid_mask]) / num_valid_pixels
    return abs_relative_difference

def calculate_squared_relative_difference(est_depth, gt_depth, valid_mask, num_valid_pixels):
    squared_relative_difference = np.sum(np.square(gt_depth[valid_mask] - est_depth[valid_mask]) / gt_depth[valid_mask]) / num_valid_pixels
    return squared_relative_difference

# TODO: avoid divide by zero
def calculate_percentage_within_threshold(est_depth, gt_depth, valid_mask, num_valid_pixels, ratio_threshold=1.25):
    # ratio = np.maximum(est_depth[valid_mask] / gt_depth[valid_mask], gt_depth[valid_mask] / est_depth[valid_mask])
    ratio = np.maximum(est_depth[valid_mask] / (gt_depth[valid_mask] + 1e-6), gt_depth[valid_mask] / (est_depth[valid_mask] + 1e-6))
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
