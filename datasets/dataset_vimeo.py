import torch.utils.data as data
import torch
from skimage.io import imread, imsave
import numpy as np
import random
from os.path import join
import glob
import h5py
import sys
import os
from os.path import join
from PIL import Image, ImageOps
import visualization as vl
from utils.file_client import FileClient
from utils.img_util import imfrombytes

#=================== Utils ===================#
def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

#=================== Testing ===================#

class DataValSetRNN(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, opt, isReal= False):
        #one input & ground truth
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir
        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]

        self.isReal = isReal
        self.frames = opt.frames
        self.frame_batch = opt.frame_batch
        self.scale = opt.scale
        self.img_num = len(self.names)
        self.world_size = opt.world_size
        self.rank_index =opt.rank_index
        self.verbose = opt.verbose

    def __len__(self):
        return 1

    def __getitem__(self, index):
        ################# For LR_Blur Images #################
        inputs = []
        flow = []
        img_num = len(self.names)
        name = self.names[index]
        dir = self.input_dir

        indexs = range(img_num)

        input_imgs = [Image.open(join(self.input_dir, self.names[i])) for i in indexs]

        for input_img in input_imgs:
            # input_img = input_img.resize((512, 512), Image.BICUBIC)
            input_patch = np.array(input_img, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            inputs.append(input_patch)


        # input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
        # size_in = input_img.size

        if not self.isReal:
        ################# For GT and LR_Deblur Images #################
            gt = []
            gt_imgs = [Image.open(os.path.join(self.gt_dir, self.names[i])).convert('RGB') for i in indexs]

            for gt_img in gt_imgs:

                # gt_img = gt_img.resize((512, 512), Image.BICUBIC)

                gt_patch = np.array(gt_img, dtype=np.float32) / 255
                gt_patch = gt_patch.transpose((2, 0, 1))
                gt.append(gt_patch)

            return np.stack(inputs,axis=0).copy(), \
                   np.stack(inputs,axis=0).copy(), \
                   np.stack(gt,axis=0).copy(),\
                   name,\
                   dir

        return np.vstack(input).copy(), name, dir

class DataValSetRNN_full(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, opt, isReal= False):
        #one input & ground truth
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir

        self.isReal = isReal
        self.frames = opt.frames
        self.frame_batch = opt.frame_batch
        self.scale = opt.scale
        self.rank_index =opt.rank_index
        self.verbose = opt.verbose

        self.keys = []
        with open(opt.testlist, 'r') as fin:
            for index, line in enumerate(fin):
                if index % 50 == 0: 
                    self.keys.append(line.strip())
        self.img_num = len(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        self.inputlists = range(1,8)

        inputs = []
        input_imgs = []     
        for i in self.inputlists:
            path = join(self.input_dir, f'{key}/im{i:01d}.png')
            input_imgs.append(Image.open(path))
  

        for input_img in input_imgs:
            # input_img = input_img.resize((512, 512), Image.BICUBIC)
            input_patch = np.array(input_img, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            inputs.append(input_patch)


        # input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
        # size_in = input_img.size

        if not self.isReal:
        ################# For GT and LR_Deblur Images #################
            gt = []
            gt_imgs = []
            for i in self.inputlists:
                path = join(self.gt_dir, f'{key}/im{i:01d}.png')
                gt_imgs.append(Image.open(path))
            
            for gt_img in gt_imgs:

                # gt_img = gt_img.resize((512, 512), Image.BICUBIC)

                gt_patch = np.array(gt_img, dtype=np.float32) / 255
                gt_patch = gt_patch.transpose((2, 0, 1))
                gt.append(gt_patch)

            return np.stack(inputs,axis=0).copy(), \
                   np.stack(inputs,axis=0).copy(), \
                   np.stack(gt,axis=0).copy(),\
                   key

        return np.vstack(input).copy(), key

#=================== Training ===================#

class DataSetRNN(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, opt):
        super(DataSetRNN, self).__init__()
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir
        self.frames = opt.frames
        self.cropSize = opt.cropSize
        self.repeat = opt.repeat
        self.frame_batch = opt.frame_batch
        self.scale = opt.scale
        self.batchSize = opt.batchSize

        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]
        self.img_num = len(self.names)
        self.dataset_len = len(self.names) // self.frame_batch

    def __len__(self):
        return self.dataset_len + self.batchSize - (self.dataset_len % self.batchSize)

    def __getitem__(self, index):
        # print(index)
        # numpy
        # scale = random.randint(10, 10)/10
        index = index * self.frame_batch
        aug_scale = random.choice([0.5, 0.75, 1])
        # aug_scale = random.choice([1])
        # index = index % (self.img_num-(self.frame_batch-1))
        # scale = 1
        # interval = random.choice([-3, -2, -1, 1, 2, 3])
        interval = 1
        # img_num = len(self.names)

        self.inputlists = np.clip(range(index,index+self.frame_batch), 0, self.img_num - 1)
        # self.inputlists = [8 , 9, 10, 11, 12, 13, 14, 15]
        # print(self.inputlists)

        self.fineSize = self.cropSize // aug_scale

        ################# For LR_Blur Images #################
        inputs = []
        flow = []

        input_imgs = [Image.open(join(self.input_dir, self.names[i])) \
                      for i in self.inputlists]

        left_top_w = random.randint(0, input_imgs[0].size[0] - self.fineSize - 1)
        left_top_h = random.randint(0, input_imgs[0].size[1] - self.fineSize - 1)
        isFilp = random.randint(0, 1)
        rotation_times = random.randint(0, 0)
        for input_img in input_imgs:
            input_patch = input_img.crop(
                (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))

            # input_patch.resize(face_size, Image.ANTIALIAS)
            input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            input_patch = np.array(input_patch, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            # randomly flip
            if isFilp == 0:
                input_patch = np.flip(input_patch, 2)
            # randomly rotation
            input_patch = np.rot90(input_patch, rotation_times, (1, 2))
            inputs.append(input_patch)

        ################# For GT and LR_Deblur Images #################
        gt = []

        gt_imgs = [Image.open(os.path.join(self.gt_dir, self.names[i])).convert('RGB') for i in self.inputlists]

        for gt_img in gt_imgs:
            gt_patch = gt_img.crop(
                (left_top_w*self.scale, left_top_h*self.scale, left_top_w*self.scale + self.fineSize*self.scale, 
                left_top_h*self.scale + self.fineSize*self.scale))
            gt_patch = gt_patch.resize((self.cropSize*self.scale, self.cropSize*self.scale), Image.BICUBIC)

            gt_patch = np.array(gt_patch, dtype=np.float32) / 255
            gt_patch = gt_patch.transpose((2, 0, 1))
            # randomly flip
            if isFilp == 0:
                gt_patch = np.flip(gt_patch, 2)
            # randomly rotation
            gt_patch = np.rot90(gt_patch, rotation_times, (1, 2))
            gt.append(gt_patch)

        return np.stack(inputs,axis=0).copy(), \
               np.stack(inputs,axis=0).copy(), \
               np.stack(gt,axis=0).copy(), self.inputlists

class DataSetRNN_full(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, opt, local_rank=0):
        super(DataSetRNN_full, self).__init__()
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir
        self.frames = opt.frames
        self.cropSize = opt.cropSize
        self.repeat = opt.repeat
        self.frame_batch = opt.frame_batch
        self.scale = opt.scale
        self.batchSize = opt.batchSize
        self.file_client = FileClient()
        self.world_size = opt.world_size
        self.rank_index =opt.rank_index
        self.verbose = opt.verbose

        self.keys = []
        with open(opt.trainlist, 'r') as fin:
            self.keys = [line.strip() for line in fin]
        # for path in sorted(os.listdir(self.gt_dir)):
        #     imgs_num = len(os.listdir(os.path.join(self.gt_dir, path)))
        #     self.keys.extend(
        #         [f'{path}/{i:08d}' for i in range(int(imgs_num-self.frame_batch))])

        # self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]
        self.img_num = len(self.keys)
        self.num_iter_per_epoch = len(self.keys) * self.repeat // self.batchSize // self.world_size
        print('Iterations per epoch is: {}'.format(self.num_iter_per_epoch))

    def __len__(self):
        return len(self.keys)* self.repeat

    def __getitem__(self, index):
        # print(index)
        # numpy
        # scale = random.randint(10, 10)/10
        # if self.rank_index == 1:
        #     print("rank 1")
        # if self.verbose:
        #     print('rank is {} and index is {}'.format(self.rank_index, index))
        index = index % len(self.keys)
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        aug_scale = random.choice([1])
        # aug_scale = random.choice([1])
        # index = index % (self.img_num-(self.frame_batch-1))
        # scale = 1
        # interval = random.choice([-3, -2, -1, 1, 2, 3])
        interval = 1
        # img_num = len(self.names)

        self.inputlists = range(1,8)
        # self.inputlists = [8 , 9, 10, 11, 12, 13, 14, 15]
        # print(self.inputlists)

        self.fineSize = self.cropSize // aug_scale

        ################# For LR_Blur Images #################
        inputs = []
        flow = []
        input_imgs = []
        for i in self.inputlists:
            path = join(self.input_dir, f'{key}/im{i:01d}.png')
            input_imgs.append(Image.open(path))


        # input_imgs = [Image.open(join(self.input_dir, f'{clip_name}/{i:08d}.png')) \
        #               for i in self.inputlists]

        # input_imgs = []       
        # for i in self.inputlists:
        #     path = join(self.input_dir, f'{clip_name}/{i:08d}.png')
        #     input_bytes = self.file_client.get(path, 'input')
        #     input_img = imfrombytes(input_bytes, float32=False)
        #     input_img = Image.fromarray(input_img[:, :, ::-1])
        #     input_imgs.append(input_img)
 
        left_top_w = random.randint(0, input_imgs[0].size[0] - self.fineSize - 1)
        left_top_h = 0
        isFilp = random.randint(0, 1)
        rotation_times = random.randint(0, 0)
        for input_img in input_imgs:
            input_patch = input_img.crop(
                (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize ))

            # input_patch.resize(face_size, Image.ANTIALIAS)
            input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            input_patch = np.array(input_patch, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            # randomly flip
            if isFilp == 0:
                input_patch = np.flip(input_patch, 2)
            # randomly rotation
            input_patch = np.rot90(input_patch, rotation_times, (1, 2))
            inputs.append(input_patch)

        ################# For GT and LR_Deblur Images #################
        gts = []
        gt_imgs = []
        for i in self.inputlists:
            path = join(self.gt_dir, f'{key}/im{i:01d}.png')
            gt_imgs.append(Image.open(path))
        # gt_imgs = [Image.open(join(self.gt_dir, f'{clip_name}/{i:08d}.png' )) \
        #         for i in self.inputlists]
    
        # gt_imgs = []       
        # for i in self.inputlists:
        #     path = join(self.gt_dir, f'{clip_name}/{i:08d}.png')
        #     gt_bytes = self.file_client.get(path, 'gt')
        #     gt_img = imfrombytes(gt_bytes, float32=False)
        #     gt_img = Image.fromarray(gt_img[:, :, ::-1])
        #     gt_imgs.append(gt_img)

        for gt_img in gt_imgs:
            gt_patch = gt_img.crop(
                (left_top_w*self.scale, left_top_h*self.scale, left_top_w*self.scale + self.fineSize*self.scale, 
                left_top_h*self.scale + self.fineSize*self.scale))
            gt_patch = gt_patch.resize((self.cropSize*self.scale, self.cropSize*self.scale), Image.BICUBIC)

            gt_patch = np.array(gt_patch, dtype=np.float32) / 255
            gt_patch = gt_patch.transpose((2, 0, 1))
            # randomly flip
            if isFilp == 0:
                gt_patch = np.flip(gt_patch, 2)
            # randomly rotation
            gt_patch = np.rot90(gt_patch, rotation_times, (1, 2))
            gts.append(gt_patch)

        return np.stack(inputs,axis=0).copy(), \
               np.stack(inputs,axis=0).copy(), \
               np.stack(gts,axis=0).copy(), self.inputlists