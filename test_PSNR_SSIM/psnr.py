#!/usr/bin/env python
import argparse
import utils
from PIL import Image
import numpy as np
import scipy.misc
import os
import pytorch_ssim
from skimage.io import imread, imsave


parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--data", type=str, default="/data1/VideoHazy_v3/Test/hazy/", help="path to load data images")
parser.add_argument("--gt", type=str, default="/data1/VideoHazy_v3/Test/gt/", help="path to load gt images")

opt = parser.parse_args()
print(opt)

folders = sorted(os.listdir(opt.data))
psnrs_all = []
for folder in folders:
    datas = utils.load_all_image(os.path.join(opt.data, folder))
    gts = utils.load_all_image(os.path.join(opt.gt, folder))

    datas.sort()
    gts.sort()

    def output_psnr_mse(img_orig, img_out):
        squared_error = np.square(img_orig - img_out)
        mse = np.mean(squared_error)
        psnr = 10 * np.log10(1.0 / mse)
        return psnr

    psnrs_folder = []
    for i in range(len(datas)):
        psnr_data = scipy.misc.fromimage(Image.open(datas[i])).astype(float)/255.0
        psnr_gt = scipy.misc.fromimage(Image.open(gts[i])).astype(float)/255.0


        psnr = output_psnr_mse(psnr_data, psnr_gt)
        print(i+1)
        print("PSNR:", psnr)
        psnrs_all.append(psnr)
        psnrs_folder.append(psnr)
    print("AVG_PSNR:", np.mean(psnrs_folder))
print("AVG_PSNR:", np.mean(psnrs_all))

"""
75 pth
rp: 6 PSNR: 22.6392712102

"""
