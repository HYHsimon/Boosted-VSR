from __future__ import print_function

import argparse
import os
import time
from math import log10
from os.path import join

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import statistics
from datasets.dataset_hf5 import DataValSet

import pytorch_ssim


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def test(test_gen, criterion):
    avg_psnr = 0
    avg_ssim = 0

    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            input ,target = batch[0].to(device), batch[1].to(device)
            img_name = batch[2]
            # print(img_name)
            # print(input.shape())

            mse = criterion(input, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            print(iteration)
            print(psnr)
            ssim = pytorch_ssim.ssim(input, target)
            print(ssim)
            avg_ssim += ssim
            
            #resultSRDeblur = transforms.ToPILImage()(input.cpu()[0])
            #resultSRDeblur.save(join(SR_dir, '{0:05d}_{1}_{2}.png'.format(iteration, 'DuRN', psnr)))


    print("===> Avg. SR PSNR: {:.4f} dB".format(avg_psnr / iteration))
    print("===> Avg. SR SSIM: {:.4f} dB".format(avg_ssim / iteration))


print("===> Loading datasets")
#root_val_dir = '/4TB1/xianglei/NYU_try_image/NYU_select/test_image/'
#root_val_dir = '/4TB1/xianglei/real_we/'
root_val_dir = '/4TB/datasets/RESIDE/xianglei/test/SOTS_test/new/'
#SR_dir = '/4TB1/CVPR20/RESIDE/DuRN_rename/'
#root_val_dir = '/4TB/datasets/RESIDE/xianglei/ICONIP/'

testloader = DataLoader(
    DataValSet(root_val_dir),
    batch_size=1, shuffle=False, pin_memory=False)


criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.to(device)

test(testloader,  criterion)

