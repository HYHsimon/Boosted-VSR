import torch.utils.data as data
import torch
from skimage.io import imread, imsave
import torchvision.transforms as tf
import numpy as np
import random
import os
import glob
import h5py
import sys
import os
from PIL import Image
sys.path.append('..')
from skimage.transform import rescale
from os.path import join
from skimage.transform import rotate
from skimage import img_as_float
import torchvision

#=================== For Testing ===================#
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPG"])

#
class DataValSet(data.Dataset):
    def __init__(self, root, mean=(128, 128, 128), isReal= False):
        self.root = root
        self.mean = mean
        self.isReal = isReal
        self.input_dir = os.path.join(self.root, 'dehazed_x4_v2')
        #self.target_lr_dir = os.path.join(self.root, 'LR')
        self.target_dir = os.path.join(self.root, 'HR')


        # for split in ["train", "trainval", "val"]:
        self.input_ids = [x for x in sorted(os.listdir(self.input_dir)) if is_image_file(x)]
       # self.target_lr_ids = [x for x in sorted(os.listdir(self.target_lr_dir)) if is_image_file(x)]
        if not self.isReal:
            self.target_ids = [x for x in sorted(os.listdir(self.target_dir)) if is_image_file(x)]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        name = self.input_ids[index]
        input_image = imread(os.path.join(self.input_dir, "%s" % name))
        #input_image = input_image[0]   ##### may need delete
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.asarray(input_image, np.float32)
        input_image /= 255

        if not self.isReal:
            name = self.target_ids[index]
            target_image = imread(os.path.join(self.target_dir, "%s" % name))
            #target_image =target_image[0] ##### may need delete
            target_image = target_image.transpose((2, 0, 1))
            target_image = np.asarray(target_image, np.float32)
            target_image /= 255

        if not self.isReal:
            return input_image.copy(), target_image.copy(), name
        else:
            return input_image.copy(), name


#=================== For Training ===================#


class DataSet_HDF5(data.Dataset):
    def __init__(self, file_path):
        super(DataSet_HDF5, self).__init__()
        hf = h5py.File(file_path,'r')
        self.data = hf.get("data")
        self.target = hf.get("label")
        # hf.close()

    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]
        LR_patch = np.clip(LR_patch, 0, 1)  # we might get out of bounds due to noise
        HR_patch = np.clip(HR_patch, 0, 1)  # we might get out of bounds due to noise
        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)

        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, 2)
            HR_patch = np.flip(HR_patch, 2)
        # randomly rotation
        rotation_degree = random.randint(0, 3)
        LR_patch = np.rot90(LR_patch, rotation_degree, (1,2))
        HR_patch = np.rot90(HR_patch, rotation_degree, (1,2))
        return LR_patch.copy(), \
               HR_patch.copy()

    def __len__(self):
        return self.data.shape[0]


class DataSet_HDF5_Multi(data.Dataset):
    def __init__(self, file_path):
        super(DataSet_HDF5_Multi, self).__init__()
        hf = h5py.File(file_path,'r')
        self.data = hf.get("data")
        self.target = hf.get("label")
        # hf.close()

    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]

        LR_patch = np.clip(LR_patch, 0, 1)  # we might get out of bounds due to noise
        HR_patch = np.clip(HR_patch, 0, 1)  # we might get out of bounds due to noise
        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)

        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, 2)
            HR_patch = np.flip(HR_patch, 2)
        # randomly rotation
        rotation_degree = random.randint(0, 3)
        LR_patch = np.rot90(LR_patch, rotation_degree, (1,2))
        HR_patch = np.rot90(HR_patch, rotation_degree, (1,2))

        GT = HR_patch
        GT_s2 = np.asarray(rescale(GT.transpose((1,2,0)), scale=1/2, anti_aliasing=True),np.float32).transpose((2,0,1))
        GT_s4 = np.asarray(rescale(GT.transpose((1,2,0)), scale=1/4, anti_aliasing=True),np.float32).transpose((2,0,1))
        GT_s8 = np.asarray(rescale(GT.transpose((1,2,0)), scale=1/8, anti_aliasing=True),np.float32).transpose((2,0,1))
        GT_s16 = np.asarray(rescale(GT.transpose((1,2,0)), scale=1/16, anti_aliasing=True),np.float32).transpose((2,0,1))
        # img_gt = Image.fromarray(np.uint8(GT.transpose((1,2,0))*255))
        # img_gt_s2 = Image.fromarray(np.uint8(GT_s8 * 255))

        return LR_patch.copy(), \
               GT.copy(), GT_s2.copy(), GT_s4.copy(), GT_s8.copy(), GT_s16.copy()

    def __len__(self):
        return self.data.shape[0]


class DataSet(data.Dataset):
    def __init__(self, input_dir, gt_dir, cropSize=256, sigma=2, gamma=2.2):
        super(DataSet, self).__init__()
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.cropSize = cropSize
        self.sigma = sigma
        self.gamma = gamma

        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]

    def __len__(self):
        return len(self.names)*8

    def __getitem__(self, index):
        #print(index)
        # numpy
        index = index%len(self.names)
        input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')

        # scale = random.choice([1])
        scale = random.choice([0.5, 0.75, 1])
        self.fineSize = self.cropSize//scale

        while input_img.size[0]<self.fineSize+2 or input_img.size[1]<self.fineSize+2:
            index = random.randint(0, len(self.names))
            input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
            gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')

        left_top_w = random.randint(0, input_img.size[0] - self.fineSize - 1)
        left_top_h = random.randint(0, input_img.size[1] - self.fineSize - 1)
        # hr_img.show()
        input_patch = input_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        gt_patch = gt_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))

        # input_patch.resize(face_size, Image.ANTIALIAS)
        input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
        gt_patch = gt_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)

        input_patch = np.array(input_patch, dtype=np.float32) / 255
        gt_patch = np.array(gt_patch, dtype=np.float32) / 255


        # #hr_patch[hr_patch > 0.8] = 0.8
        # hr_patch = hr_patch**self.gamma / 2
        # noise_patch = hr_patch / random.uniform(3, 5)
        # input_patch = (noise_patch + np.random.normal(0, self.sigma / 255.0, noise_patch.shape)).astype(np.float32)
        # np.clip(input_patch, 0, 1, out=input_patch)

        input_patch = input_patch.transpose((2, 0, 1))
        gt_patch = gt_patch.transpose((2, 0, 1))

        # randomly flip
        if random.randint(0, 1) == 0:
            input_patch  = np.flip(input_patch, 2)
            gt_patch     = np.flip(gt_patch, 2)
        # randomly rotation
        rotation_times = random.randint(0, 3)
        input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
        gt_patch       = np.rot90(gt_patch, rotation_times, (1, 2))

        return input_patch.copy(), \
               gt_patch.copy()