import os
import random
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.image as img
import scipy.io as sio
import re
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from torch.nn import functional as F
from scipy.ndimage import measurements, interpolation
from scipy import signal
from imresize import imresize
from blur_downsample import generate_gaussian_kernel

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel

def kernel_injection(hr, kernel_path):
    # kernel_Path = 'Results/DPEDiphone-tr-x1'
    kernel_Path = kernel_path
    kernel_size = 13
    scale = 2.0
    noise_std = 0
    if kernel_Path:
        kernel_pool = sorted_aphanumeric(os.listdir(kernel_Path))
        ###############CVPRW downsampling
        kernel_idx = random.randint(0, len(kernel_pool)-1)
        kernel = sio.loadmat(os.path.join(kernel_Path, kernel_pool[kernel_idx], '%s_kernel_x2.mat'%(kernel_pool[kernel_idx])))['Kernel']
    else:
        kernel = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    # Real_Img = img.imread(os.path.join(Real_Path, filename))
    HR_patch = hr
    kernel_post = kernel_shift(kernel, [scale, scale])
    LR_patch = imresize(HR_patch, 1.0 / (np.ones(2)*scale), kernel=kernel)
    # LR_patch = np.clip(LR_patch + np.random.randn(*LR_patch.shape) *noise_std, 0, 1)

    return LR_patch




