
import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F
from scipy.ndimage import measurements, interpolation
from bicubic_pytorch import core
import random

def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)

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

def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = kernel_shift(gaussian_filter, [2.0, 2.0])
    pad_w, pad_h = gaussian_filter.shape[0] // 2 + scale * 2, gaussian_filter.shape[0] // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
        0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    # x = F.interpolate(x, scale_factor=2, mode='bicubic')
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x


def duf_downsample1(x, kernel_size=13, scale_ori=2):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """ 
    scale = scale_ori * 2
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)

    # gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    # gaussian_filter = kernel_shift(gaussian_filter, [scale, scale])
    # pad_w, pad_h = gaussian_filter.shape[0] // 2 + scale * 2, gaussian_filter.shape[0] // 2 + scale * 2
    # x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    # gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
    #     0).unsqueeze(0)
    # x = F.conv2d(x, gaussian_filter, stride=scale)
    # x = x[:, :, 2:-2, 2:-2]
    x_resized_x4 = core.imresize(x, scale=0.25)
    x_resized_x2 = core.imresize(x_resized_x4, scale=2)
    # x = F.interpolate(x, scale_factor=0.25, mode='bicubic')
    # x = F.interpolate(x, scale_factor=scale_ori, mode='bicubic')
    x = torch.clamp(x_resized_x2 , 0, 1)
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x
    
def duf_downsample2(x, kernel_size=13, scale_ori=2):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    scale = scale_ori * 2
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = kernel_shift(gaussian_filter, [scale, scale])
    pad_w, pad_h = gaussian_filter.shape[0] // 2 + scale * 2, gaussian_filter.shape[0] // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
        0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = F.interpolate(x, scale_factor=scale_ori, mode='bicubic')
    x = torch.clamp(x , 0, 1)
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x

def duf_downsample3(x, kernel_size=13, scale_ori=2):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """ 
    scale = scale_ori * 2
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)

    # gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    # gaussian_filter = kernel_shift(gaussian_filter, [scale, scale])
    # pad_w, pad_h = gaussian_filter.shape[0] // 2 + scale * 2, gaussian_filter.shape[0] // 2 + scale * 2
    # x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    # gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
    #     0).unsqueeze(0)
    # x = F.conv2d(x, gaussian_filter, stride=scale)
    # x = x[:, :, 2:-2, 2:-2]
    x_resized_x2 = core.imresize(x, scale=0.5)
    # x = F.interpolate(x, scale_factor=0.25, mode='bicubic')
    # x = F.interpolate(x, scale_factor=scale_ori, mode='bicubic')
    x = torch.clamp(x_resized_x2 , 0, 1)
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x



def duf_downsample5(x, kernel_size=13, scale=2):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """ 
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)

    # gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    # gaussian_filter = kernel_shift(gaussian_filter, [scale, scale])
    # pad_w, pad_h = gaussian_filter.shape[0] // 2 + scale * 2, gaussian_filter.shape[0] // 2 + scale * 2
    # x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    # gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
    #     0).unsqueeze(0)
    # x = F.conv2d(x, gaussian_filter, stride=scale)
    # x = x[:, :, 2:-2, 2:-2]
    random_scale = [2,3,4]
    s = random.choice(random_scale)
    x_resized_down = core.imresize(x, scale= 1/s)
    x_resized_up = core.imresize(x_resized_down, scale=s)

    x_resized_x2 = core.imresize(x_resized_up, scale=0.5)

    x = torch.clamp(x_resized_x2 , 0, 1)
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x