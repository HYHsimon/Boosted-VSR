import torch.utils.data as data
import numpy as np
import random
import os
from os.path import join
from PIL import Image
from utils.file_client import FileClient
from utils.img_util import imfrombytes

#=================== Utils ===================#
def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

#=================== Testing ===================#

class DataValSetRNN(data.Dataset):
    def __init__(self, input_dir, gt_dir, opt):
        #one input & ground truth
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.names_GT = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]
        self.names_LR = [x for x in sorted(os.listdir(self.input_dir)) if is_image_file(x)]

        self.isReal = opt.isReal
        self.frames = opt.frames
        self.frame_batch_test = opt.frame_batch_test
        self.scale = opt.scale
        self.img_num = len(self.names_LR)
        self.verbose = opt.verbose

        if self.frame_batch_test == -1:
            self.dataset_len = 1
        else:
            if self.img_num % self.frame_batch_test == 0:
                self.dataset_len = self.img_num // self.frame_batch_test
            else:
                self.dataset_len = self.img_num // self.frame_batch_test + 1


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        ################# For LR_Blur Images #################
        inputs = []
        img_num = len(self.names_LR)
        
        dir = self.input_dir

        if self.frame_batch_test == -1:
            indexs = range(img_num)
        else:
            start_num = index * self.frame_batch_test
            end_num = np.clip((index + 1) * self.frame_batch_test, 0, self.img_num) 
            indexs = range(start_num, end_num)

        input_imgs = [Image.open(join(self.input_dir, self.names_LR[i])) for i in indexs]

        for input_img in input_imgs:

            input_patch = np.array(input_img, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            inputs.append(input_patch)

        if not self.isReal:
        ################# For GT and LR_Deblur Images #################
            gt = []
            gt_imgs = [Image.open(os.path.join(self.gt_dir, self.names_GT[i])).convert('RGB') for i in indexs]

            for gt_img in gt_imgs:

                gt_patch = np.array(gt_img, dtype=np.float32) / 255
                gt_patch = gt_patch.transpose((2, 0, 1))
                gt.append(gt_patch)

            return np.stack(inputs, axis=0).copy(), \
               np.stack(gt,axis=0).copy(),\
               self.names_LR,\
               dir

        return np.stack(inputs, axis=0).copy(), self.names_LR, dir

class DataValSetRNN_full(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, opt, isReal= False):
        # one input & ground truth
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir
        self.names_GT = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]
        self.names_LR = [x for x in sorted(os.listdir(self.input_dir)) if is_image_file(x)]

        self.isReal = isReal
        self.frames = opt.frames
        self.frame_batch_test = opt.frame_batch_test
        self.scale = opt.scale
        self.img_num = len(self.names_LR)
        self.verbose = opt.verbose

        if self.frame_batch_test == -1:
            self.dataset_len = 1
        else:
            self.dataset_len = self.img_num - self.frame_batch_test + 1


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        ################# For LR_Blur Images #################
        inputs = []
        img_num = len(self.names_LR)

        dir = self.input_dir

        if self.frame_batch_test == -1:
            indexs = range(img_num)
        else:
            start_num = index
            end_num = np.clip(index + self.frame_batch_test, 0, self.img_num)
            indexs = range(start_num, end_num)

        input_imgs = [Image.open(join(self.input_dir, self.names_LR[i])) for i in indexs]

        for input_img in input_imgs:

            input_patch = np.array(input_img, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            inputs.append(input_patch)

        if not self.isReal:
            ################# For GT and LR_Deblur Images #################
            gt = []
            gt_imgs = [Image.open(os.path.join(self.gt_dir, self.names_GT[i])).convert('RGB') for i in indexs]

            for gt_img in gt_imgs:

                gt_patch = np.array(gt_img, dtype=np.float32) / 255
                gt_patch = gt_patch.transpose((2, 0, 1))
                gt.append(gt_patch)

            return np.stack(inputs, axis=0).copy(), \
                   np.stack(gt, axis=0).copy(), \
                   self.names_LR, \
                   dir

        return np.stack(inputs, axis=0).copy(), self.names_LR, dir

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

        index = index * self.frame_batch
        aug_scale = random.choice([0.5, 0.75, 1])

        self.inputlists = np.clip(range(index,index+self.frame_batch), 0, self.img_num - 1)

        self.fineSize = self.cropSize // aug_scale

        ################# For LR_Blur Images #################
        inputs = []

        input_imgs = [Image.open(join(self.input_dir, self.names[i])) \
                      for i in self.inputlists]

        left_top_w = random.randint(0, input_imgs[0].size[0] - self.fineSize - 1)
        left_top_h = random.randint(0, input_imgs[0].size[1] - self.fineSize - 1)
        isFilp = random.randint(0, 1)
        rotation_times = random.randint(0, 0)
        for input_img in input_imgs:
            input_patch = input_img.crop(
                (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))

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
        self.rank_index = opt.rank_index
        self.verbose = opt.verbose

        self.keys = []
        for path in sorted(os.listdir(self.gt_dir)):
            imgs_num = len(os.listdir(os.path.join(self.gt_dir, path)))

            if self.frame_batch == -1 :
                self.frame_batch = imgs_num

            self.keys.extend(
                [f'{path}/{i:08d}' for i in range(int(imgs_num - self.frame_batch) + 1)])

        # self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]
        self.img_num = len(self.keys)
        self.num_iter_per_epoch = len(self.keys) * self.repeat // self.batchSize // self.world_size
        print('Iterations per epoch is: {}'.format(self.num_iter_per_epoch))

    def __len__(self):
        return len(self.keys) * self.repeat

    def __getitem__(self, index):

        index = index % len(self.keys)
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        aug_scale = random.choice([1])

        self.inputlists = np.clip(range(int(frame_name), int(frame_name) + self.frame_batch), 0, self.img_num - 1)

        self.fineSize = self.cropSize // aug_scale

        ################# For LR_Blur Images #################
        inputs = []

        input_imgs = []
        for i in self.inputlists:
            path = join(self.input_dir, f'{clip_name}/{i:08d}.png')
            input_bytes = self.file_client.get(path, 'input')
            input_img = imfrombytes(input_bytes, float32=False)
            input_img = Image.fromarray(input_img[:, :, ::-1])
            input_imgs.append(input_img)

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
        gts = []

        gt_imgs = []
        for i in self.inputlists:
            path = join(self.gt_dir, f'{clip_name}/{i:08d}.png')
            gt_bytes = self.file_client.get(path, 'gt')
            gt_img = imfrombytes(gt_bytes, float32=False)
            gt_img = Image.fromarray(gt_img[:, :, ::-1])
            gt_imgs.append(gt_img)

        for gt_img in gt_imgs:
            gt_patch = gt_img.crop(
                (left_top_w * self.scale, left_top_h * self.scale, left_top_w * self.scale + self.fineSize * self.scale,
                 left_top_h * self.scale + self.fineSize * self.scale))
            gt_patch = gt_patch.resize((self.cropSize * self.scale, self.cropSize * self.scale), Image.BICUBIC)

            gt_patch = np.array(gt_patch, dtype=np.float32) / 255
            gt_patch = gt_patch.transpose((2, 0, 1))
            # randomly flip
            if isFilp == 0:
                gt_patch = np.flip(gt_patch, 2)
            # randomly rotation
            gt_patch = np.rot90(gt_patch, rotation_times, (1, 2))
            gts.append(gt_patch)

        return np.stack(inputs, axis=0).copy(), \
               np.stack(inputs, axis=0).copy(), \
               np.stack(gts, axis=0).copy(), self.inputlists