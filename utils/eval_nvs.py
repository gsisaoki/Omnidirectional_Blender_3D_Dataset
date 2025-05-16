# This evaluation script is based on the original evaluation script from 3DGS (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# We are following the license of the original 3DGS codebase.

import lpips
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def read_image(render_path, gt_path):
    
    render_path = Path(render_path)
    gt_path = Path(gt_path)
    
    render = Image.open(render_path)
    gt = Image.open(gt_path)
    
    render_tensor = tf.to_tensor(render).unsqueeze(0)[:3, :, :].cuda()
    gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:3, :, :].cuda()
        
    return render_tensor, gt_tensor

def calculate_metrics(render, gt, device='cuda:0'):
    
    __LPIPS__ = {'alex': lpips.LPIPS(net='alex', version='0.1').eval().to(device), 'vgg': lpips.LPIPS(net='vgg', version='0.1').eval().to(device)}
    
    ssim_render = ssim(render, gt).item()
    psnr_render = psnr(render, gt).item()
    lpipsA_render = __LPIPS__['alex'](gt, render, normalize=True).item()
    lpipsV_render = __LPIPS__['vgg'](gt, render, normalize=True).item()
    
    metrics = {
        "PSNR": psnr_render,
        "SSIM": ssim_render,
        "LPIPS(alex)": lpipsA_render,
        "LPIPS(vgg)": lpipsV_render
    }
    
    return metrics

# if __name__ == "__main__":
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)

#     gt_path = "/path/to/gt/00000.png"
#     render_path = "/path/to/renders/00000.png"
    
#     render, gt = read_image(render_path, gt_path)
#     metrics = calculate_metrics(render, gt)
    
#     print("---------------------------------")
#     print("  PSNR        : {:>12.7f}".format(metrics['PSNR'], ".5"))
#     print("  SSIM        : {:>12.7f}".format(metrics['SSIM'], ".5"))
#     print("  LPIPS(alex) : {:>12.7f}".format(metrics['LPIPS(alex)'], ".5"))
#     print("  LPIPS(vgg)  : {:>12.7f}".format(metrics['LPIPS(vgg)'], ".5"))
#     print("---------------------------------")