import os
import glob
from PIL import Image
from torchvision import transforms
from blur_downsample import *
import torch
import numpy as np
from kernel_injection import kernel_injection
import random

def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

mode = 'BD5'
mode_name = 'BD5'
print(mode)
gt_root = '/opt/tiger/Datasets/VSR/Youku/GT_compress/frames'
save_root = '/opt/tiger/Datasets/VSR/Youku/LR/frames'

if mode == 'BD':
    print(mode)
    scale = 2
    kernel_size = 13
    save_root = os.path.join(save_root, mode_name)
    folders = sorted(os.listdir(gt_root))
    for folder in folders:
        print(folder)
        frames = []
        names = sorted(os.listdir(os.path.join(gt_root, folder)))
        for name in names:
            if is_image_file(name):
                frame_path = os.path.join(gt_root, folder, name)
                frame = Image.open(frame_path)
                frames.append(transforms.ToTensor()(frame))
        GTs =  torch.stack(frames, dim=0) 
        t, c, h, w = GTs.shape
        LRs = duf_downsample(GTs, kernel_size=13, scale=scale)
        for index in range(t):
            LR = LRs[index]
            LR = transforms.ToPILImage()(LR) 
            save_path = os.path.join(save_root, folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            LR_path = os.path.join(save_path, names[index][:-4]+'.png')
            LR.save(LR_path)
if mode == 'BD1':
    kernel_size = 13
    scale = 2
    save_root = os.path.join(save_root, mode_name)
    folders = sorted(os.listdir(gt_root))
    for folder in folders:
        print(folder)
        frames = []
        names = sorted(os.listdir(os.path.join(gt_root, folder)))
        for name in names:
            if is_image_file(name):
                frame_path = os.path.join(gt_root, folder, name)
                frame = Image.open(frame_path)
                frames.append(transforms.ToTensor()(frame))
        GTs =  torch.stack(frames, dim=0) 
        t, c, h, w = GTs.shape
        LRs = duf_downsample1(GTs, kernel_size=13, scale_ori=scale)
        for index in range(t):
            LR = LRs[index]
            LR = transforms.ToPILImage()(LR) 
            save_path = os.path.join(save_root, folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            LR_path = os.path.join(save_path, names[index][:-4]+'.png')
            LR.save(LR_path)
elif mode == 'BD2':
    kernel_size = 13
    scale = 2
    save_root = os.path.join(save_root, mode_name)
    folders = sorted(os.listdir(gt_root))
    for folder in folders:
        print(folder)
        frames = []
        names = sorted(os.listdir(os.path.join(gt_root, folder)))
        for name in names:
            if is_image_file(name):
                frame_path = os.path.join(gt_root, folder, name)
                frame = Image.open(frame_path)
                frames.append(transforms.ToTensor()(frame))
        GTs =  torch.stack(frames, dim=0) 
        t, c, h, w = GTs.shape
        LRs = duf_downsample2(GTs, kernel_size=13, scale_ori=scale)
        for index in range(t):
            LR = LRs[index]
            LR = transforms.ToPILImage()(LR) 
            save_path = os.path.join(save_root, folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            LR_path = os.path.join(save_path, names[index][:-4]+'.png')
            LR.save(LR_path)
elif mode == 'BD3':
    scale = 2
    kernel_size = 13
    save_root = os.path.join(save_root, mode_name)
    folders = sorted(os.listdir(gt_root))
    for folder in folders:
        print(folder)
        frames = []
        names = sorted(os.listdir(os.path.join(gt_root, folder)))
        for name in names:
            if is_image_file(name):
                frame_path = os.path.join(gt_root, folder, name)
                frame = Image.open(frame_path)
                frames.append(transforms.ToTensor()(frame))
        GTs =  torch.stack(frames, dim=0) 
        t, c, h, w = GTs.shape
        LRs = duf_downsample(GTs, kernel_size=13, scale=scale)
        for index in range(t):
            LR = LRs[index]
            LR = transforms.ToPILImage()(LR) 
            save_path = os.path.join(save_root, folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            LR_path = os.path.join(save_path, names[index][:-4]+'.png')
            LR.save(LR_path)
elif mode == 'BD5':
    scale = 2
    kernel_size = 13
    save_root = os.path.join(save_root, mode_name)
    folders = sorted(os.listdir(gt_root))
    for folder in folders:
        print(folder)
        frames = []
        names = sorted(os.listdir(os.path.join(gt_root, folder)))
        for name in names:
            if is_image_file(name):
                frame_path = os.path.join(gt_root, folder, name)
                frame = Image.open(frame_path)
                frames.append(transforms.ToTensor()(frame))
        GTs =  torch.stack(frames, dim=0) 
        t, c, h, w = GTs.shape
        LRs = duf_downsample5(GTs, kernel_size=13, scale=scale)
        for index in range(t):
            LR = LRs[index]
            LR = transforms.ToPILImage()(LR) 
            save_path = os.path.join(save_root, folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            LR_path = os.path.join(save_path, names[index][:-4]+'.jpg')
            q = random.randint(50,80)
            LR.save(LR_path, quality=q)
elif mode == 'injection':
    # For shifting the kernel
    cropSize = 256
    scale = 2.0
    noise_std = 0
    save_root = os.path.join(save_root, mode)
    folders = sorted(os.listdir(gt_root))
    for folder in folders:

        save_path = os.path.join(save_root, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        names = sorted(os.listdir(os.path.join(gt_root, folder)))
        for name in names:
            if is_image_file(name):
                frame_path = os.path.join(gt_root, folder, name)
                HR_Img = Image.open(frame_path)
                HR_patch = np.array(HR_Img, dtype=np.uint8) / 255.
                LR_Patch = kernel_injection(HR_patch, None)
                LR_path = os.path.join(save_path, name)
                Image.fromarray((LR_Patch  * 255).astype('uint8')).save(LR_path)






