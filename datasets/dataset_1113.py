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

#=================== Utils ===================#
def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

#=================== Testing ===================#

class DataValSet(data.Dataset):
    def __init__(self, root_dir, isReal= False):
        #one input & ground truth
        self.input_dir = join(root_dir, 'hazy')
        self.gt_dir = join(root_dir, 'gt')
        self.isReal = isReal
        folders = sorted(os.listdir(self.input_dir))
        self.names = []
        for folder in folders:
            for x in sorted(os.listdir(join(self.input_dir, folder))):
                self.names.append(join(folder, x))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
        size_in = input_img.size
        new_size_in = tuple([int(x * 4) for x in size_in])
        bicubic_img = input_img.resize(new_size_in, Image.BICUBIC)
        gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')

        input = np.array(input_img, dtype=np.float32).transpose((2, 0, 1)).copy() / 255
        bicubic = np.array(bicubic_img, dtype=np.float32).transpose((2, 0, 1)).copy() / 255

        if not self.isReal:
            target    = np.array(gt_img, dtype=np.float32).transpose((2, 0, 1)).copy() / 255
            return input, bicubic, target, name
        return input, bicubic, name

class DataValSetV(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, frames=5, isReal= False):
        #one input & ground truth
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir
        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]

        self.isReal = isReal
        self.frames = frames

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        ################# For LR_Blur Images #################
        input = []
        flow = []
        img_num = len(self.names)
        name = self.names[index]

        indexs = np.clip(range(index - self.frames//2, index + self.frames//2 + 1), 0, img_num-1)

        input_imgs = [Image.open(join(self.input_dir, self.names[i])) for i in indexs]

        for input_img in input_imgs:
            # input_img = input_img.resize((512, 512), Image.BICUBIC)
            input_patch = np.array(input_img, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            input.append(input_patch)


        # input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
        # size_in = input_img.size

        if not self.isReal:
        ################# For GT and LR_Deblur Images #################
            gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')
            # gt_img = gt_img.resize((512, 512), Image.BICUBIC)
            gt_patch = np.array(gt_img, dtype=np.float32) / 255
            gt_patch = gt_patch.transpose((2, 0, 1))

            return np.vstack(input).copy(), \
                   np.vstack(input).copy(), \
                   gt_patch.copy(),\
                   name

        return np.vstack(input).copy(), name
#=================== Training ===================#
def augment(img_in, img_lr, img_bic, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_lr = ImageOps.flip(img_lr)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_lr = ImageOps.mirror(img_lr)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_lr = img_lr.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_lr, img_bic, img_tar, info_aug

class DataSet(data.Dataset):
    def __init__(self, input_dir, lr_dir, gt_dir, cropSize=32, sigma=2, gamma=2.2):
        super(DataSet, self).__init__()
        self.input_dir = input_dir
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.cropSize = cropSize
        self.sigma = sigma
        self.gamma = gamma

        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        #print(index)
        # numpy
        index = index
        input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.names[index])).convert('RGB')
        size_in = input_img.size
        new_size_in = tuple([int(x * 4) for x in size_in])
        bicubic_img = input_img.resize(new_size_in, Image.BICUBIC)
        gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')

        scale = random.choice([0.5, 0.75, 1])
        # scale = random.choice([0.5, 0.75, 1])
        self.fineSize = self.cropSize//scale

        left_top_w = random.randint(0, input_img.size[0] - self.fineSize - 1)
        left_top_h = random.randint(0, input_img.size[1] - self.fineSize - 1)
        # hr_img.show()
        input_patch = input_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        lr_patch = lr_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        bicubic_patch = bicubic_img.crop(
            (left_top_w*4, left_top_h*4, left_top_w*4 + self.fineSize*4, left_top_h*4 + self.fineSize*4))
        gt_patch = gt_img.crop(
            (left_top_w*4, left_top_h*4, left_top_w*4 + self.fineSize*4, left_top_h*4 + self.fineSize*4))

        # input_patch.resize(face_size, Image.ANTIALIAS)
        input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
        lr_patch = lr_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
        bicubic_patch = bicubic_patch.resize((self.cropSize * 4, self.cropSize * 4), Image.BICUBIC)
        gt_patch = gt_patch.resize((self.cropSize*4, self.cropSize*4), Image.BICUBIC)

        input_patch, lr_patch, bicubic_patch, gt_patch, _ = augment(input_patch, lr_patch, bicubic_patch, gt_patch)

        input_patch = np.array(input_patch, dtype=np.float32) / 255
        lr_patch = np.array(lr_patch, dtype=np.float32) / 255
        bicubic_patch = np.array(bicubic_patch, dtype=np.float32) / 255
        gt_patch = np.array(gt_patch, dtype=np.float32) / 255

        # #hr_patch[hr_patch > 0.8] = 0.8
        # hr_patch = hr_patch**self.gamma / 2
        # noise_patch = hr_patch / random.uniform(3, 5)
        # input_patch = (noise_patch + np.random.normal(0, self.sigma / 255.0, noise_patch.shape)).astype(np.float32)
        # np.clip(input_patch, 0, 1, out=input_patch)

        input_patch = input_patch.transpose((2, 0, 1))
        lr_patch = lr_patch.transpose((2, 0, 1))
        bicubic_patch = bicubic_patch.transpose((2, 0, 1))
        gt_patch = gt_patch.transpose((2, 0, 1))

        return input_patch.copy(), \
               lr_patch.copy(), \
               gt_patch.copy(), \
               bicubic_patch.copy()

class DataSet_shuffle(data.Dataset):
    def __init__(self, input_dir, lr_dir, gt_dir, cropSize=40, sigma=2, gamma=2.2):
        super(DataSet_shuffle, self).__init__()
        self.selected_training_folder = [5, 7, 8, 10, 11, 17, 18, 21, 22, 25, 26, 27, 159, 66, 71, 72, 74, 75, 76, 77, 83, 90, 91, 85, 86, 87,
                       123, 124, 104, 115, 150, 139, 142, 144, 155, 166, 167, 169, 172, 176, 186, 187, 191, 192, 196, 198,
                       185, 214, 215, 216, 218, 219, 227, 228, 232, 233, 234, 235, 236, 237, 238, 239]
        self.input_dir = input_dir
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.cropSize = cropSize
        self.sigma = sigma
        self.gamma = gamma

        folders = sorted(os.listdir(self.input_dir))
        self.names = []
        for folder in folders:
            if int(folder) not in self.selected_training_folder:
                continue
            for x in sorted(os.listdir(join(self.input_dir, folder))):
                self.names.append(join(folder, x))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        #print(index)
        # numpy
        input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_dir, self.names[index])).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')

        # scale = 1
        scale = random.choice([0.5, 0.75, 1])
        self.fineSize = self.cropSize//scale

        left_top_w = random.randrange(0, input_img.size[0] - self.fineSize + 1)
        left_top_h = random.randrange(0, input_img.size[1] - self.fineSize + 1)
        # hr_img.show()
        input_patch = input_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        lr_patch = lr_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        gt_patch = gt_img.crop(
            (left_top_w*4, left_top_h*4, left_top_w*4 + self.fineSize*4, left_top_h*4 + self.fineSize*4))

        # input_patch.resize(face_size, Image.ANTIALIAS)
        if scale != 1:
            input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            gt_patch = gt_patch.resize((self.cropSize*4, self.cropSize*4), Image.BICUBIC)
            lr_patch = lr_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)

        input_patch = np.array(input_patch, dtype=np.float32) / 255
        gt_patch = np.array(gt_patch, dtype=np.float32) / 255
        lr_patch = np.array(lr_patch, dtype=np.float32) / 255


        # #hr_patch[hr_patch > 0.8] = 0.8
        # hr_patch = hr_patch**self.gamma / 2
        # noise_patch = hr_patch / random.uniform(3, 5)
        # input_patch = (noise_patch + np.random.normal(0, self.sigma / 255.0, noise_patch.shape)).astype(np.float32)
        # np.clip(input_patch, 0, 1, out=input_patch)

        input_patch = input_patch.transpose((2, 0, 1))
        gt_patch = gt_patch.transpose((2, 0, 1))
        lr_patch = lr_patch.transpose((2, 0, 1))

        # randomly flip
        if random.randint(0, 1) == 0:
            input_patch  = np.flip(input_patch, 2)
            gt_patch     = np.flip(gt_patch, 2)
            lr_patch = np.flip(lr_patch, 2)
        # randomly rotation
        rotation_times = random.randint(0, 3)
        input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
        gt_patch       = np.rot90(gt_patch, rotation_times, (1, 2))
        lr_patch = np.rot90(lr_patch, rotation_times, (1, 2))

        return input_patch.copy(), \
               lr_patch.copy(),\
               gt_patch.copy()


class DataSetV(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, frames=5, cropSize=32, sigma=2, gamma=2.2, repeat = 1):
        super(DataSetV, self).__init__()
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir
        self.cropSize = cropSize
        self.sigma = sigma
        self.gamma = gamma
        self.frames = frames
        self.repeat = repeat

        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]
        self.img_num = len(self.names)


    def __len__(self):
        return len(self.names)*self.repeat

    def __getitem__(self, index):
        #print(index)
        # numpy
        # scale = random.randint(10, 10)/10
        aug_scale = random.choice([0.5, 0.75, 1])
        # aug_scale = random.choice([1])
        index = index % self.img_num
        #scale = 1
        #interval = random.choice([-3, -2, -1, 1, 2, 3])
        interval = 1
        # img_num = len(self.names)
        if self.frames == 1:
            self.inputlists =[index]
        if self.frames == 3:
            self.inputlists = np.clip([index-interval, index, index+interval],0,self.img_num-1)
        if self.frames == 5:
            self.inputlists = np.clip([index-interval*2, index-interval, index, index+interval, index+interval*2],0,self.img_num-1)
        if self.frames == 7:
            self.inputlists = np.clip([index-interval*3, index-interval*2, index-interval, index, index+interval, index+interval*2, index+interval*3],0,self.img_num-1)
        self.fineSize = self.cropSize//aug_scale

        ################# For LR_Blur Images #################
        input = []
        flow = []

        input_imgs = [Image.open(join(self.input_dir, self.names[i]))\
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
                input_patch  = np.flip(input_patch, 2)
            # randomly rotation
            input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
            input.append(input_patch)


        ################# For GT and LR_Deblur Images #################
        gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')

        gt_patch = gt_img.crop(
            (left_top_w, left_top_h, left_top_w+ self.fineSize, left_top_h + self.fineSize))
        gt_patch = gt_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)

        gt_patch = np.array(gt_patch, dtype=np.float32) / 255
        gt_patch = gt_patch.transpose((2, 0, 1))
        # randomly flip
        if isFilp == 0:
            gt_patch     = np.flip(gt_patch, 2)
        # randomly rotation
        gt_patch       = np.rot90(gt_patch, rotation_times, (1, 2))


        return np.vstack(input).copy(), \
               np.vstack(input).copy(), \
               gt_patch.copy(), \

class DataSetV_MultiGT(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, frames=5, cropSize=32, sigma=2, gamma=2.2, repeat = 1):
        super(DataSetV_MultiGT, self).__init__()
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir
        self.cropSize = cropSize
        self.sigma = sigma
        self.gamma = gamma
        self.frames = frames
        self.repeat = repeat

        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]
        self.img_num = len(self.names)


    def __len__(self):
        return len(self.names)*self.repeat

    def __getitem__(self, index):
        #print(index)
        # numpy
        # scale = random.randint(10, 10)/10
        aug_scale = random.choice([0.5, 0.75, 1])
        # aug_scale = random.choice([1])
        index = index % self.img_num
        #scale = 1
        #interval = random.choice([-3, -2, -1, 1, 2, 3])
        interval = 1
        # img_num = len(self.names)
        if self.frames == 1:
            self.inputlists =[index]
        if self.frames == 3:
            self.inputlists = np.clip([index-interval, index, index+interval],0,self.img_num-1)
        if self.frames == 5:
            self.inputlists = np.clip([index-interval*2, index-interval, index, index+interval, index+interval*2],0,self.img_num-1)
        if self.frames == 7:
            self.inputlists = np.clip([index-interval*3, index-interval*2, index-interval, index, index+interval, index+interval*2, index+interval*3],0,self.img_num-1)
        self.fineSize = self.cropSize//aug_scale

        ################# For LR_Blur Images #################
        input = []
        gt = []
        flow = []

        input_imgs = [Image.open(join(self.input_dir, self.names[i]))\
                      for i in self.inputlists]
        gt_imgs = [Image.open(join(self.gt_dir, self.names[i]))\
                      for i in self.inputlists]

        left_top_w = random.randint(0, input_imgs[0].size[0] - self.fineSize - 1)
        left_top_h = random.randint(0, input_imgs[0].size[1] - self.fineSize - 1)
        isFilp = random.randint(0, 1)
        rotation_times = random.randint(0, 0)
        for input_img, gt_img in zip(input_imgs, gt_imgs):
            input_patch = input_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
            gt_patch = gt_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))

            # input_patch.resize(face_size, Image.ANTIALIAS)
            input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            input_patch = np.array(input_patch, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            gt_patch = gt_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            gt_patch = np.array(gt_patch, dtype=np.float32) / 255
            gt_patch = gt_patch.transpose((2, 0, 1))

            # randomly flip
            if isFilp == 0:
                input_patch  = np.flip(input_patch, 2)
                gt_patch = np.flip(gt_patch, 2)
            # randomly rotation
            input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
            gt_patch = np.rot90(gt_patch, rotation_times, (1, 2))

            input.append(input_patch)
            gt.append(gt_patch)

        return np.vstack(input).copy(), \
               np.vstack(input).copy(), \
               np.vstack(gt).copy(),

class DataSetV_Flow(data.Dataset):
    def __init__(self, input_dir, flow_dir, gt_dir, frames=5, cropSize=32, sigma=2, gamma=2.2, repeat = 1):
        super(DataSetV, self).__init__()
        self.input_dir = input_dir
        self.flow_dir = flow_dir
        self.gt_dir = gt_dir
        self.cropSize = cropSize
        self.sigma = sigma
        self.gamma = gamma
        self.frames = frames
        self.repeat = repeat

        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]
        self.img_num = len(self.names)


    def __len__(self):
        return len(self.names)*self.repeat

    def __getitem__(self, index):
        #print(index)
        # numpy
        # scale = random.randint(10, 10)/10
        aug_scale = random.choice([0.5, 0.75, 1])
        index = index % self.img_num
        #scale = 1
        #interval = random.choice([-3, -2, -1, 1, 2, 3])
        interval = 1
        # img_num = len(self.names)
        if self.frames == 1:
            self.inputlists =[index]
        if self.frames == 5:
            self.inputlists = np.clip([index-interval*2, index-interval, index, index+interval, index+interval*2],0,self.img_num-1)
        if self.frames == 7:
            self.inputlists = np.clip([index-interval*3, index-interval*2, index-interval, index, index+interval, index+interval*2, index+interval*3],0,self.img_num-1)
        self.fineSize = self.cropSize//aug_scale

        ################# For LR_Blur Images #################
        input = []
        flow = []

        input_imgs = [Image.open(join(self.input_dir, self.names[i]))\
                      for i in self.inputlists]
        flow_imgs = [Image.open(join(self.flow_dir, self.names[i]))\
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
                input_patch  = np.flip(input_patch, 2)
            # randomly rotation
            input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
            input.append(input_patch)

        for flow_img in flow_imgs:
            flow_patch = flow_img.crop(
                (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))

            # input_patch.resize(face_size, Image.ANTIALIAS)
            flow_patch = flow_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            flow_patch = np.array(flow_patch, dtype=np.float32) / 255
            flow_patch = flow_patch.transpose((2, 0, 1))
            # randomly flip
            if isFilp == 0:
                flow_patch  = np.flip(flow_patch, 2)
            # randomly rotation
            flow_patch    = np.rot90(flow_patch, rotation_times, (1, 2))
            flow.append(flow_patch)

        ################# For GT and LR_Deblur Images #################
        gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')

        gt_patch = gt_img.crop(
            (left_top_w, left_top_h, left_top_w+ self.fineSize, left_top_h + self.fineSize))
        gt_patch = gt_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)

        gt_patch = np.array(gt_patch, dtype=np.float32) / 255
        gt_patch = gt_patch.transpose((2, 0, 1))
        # randomly flip
        if isFilp == 0:
            gt_patch     = np.flip(gt_patch, 2)
        # randomly rotation
        gt_patch       = np.rot90(gt_patch, rotation_times, (1, 2))


        return np.vstack(input).copy(), \
               np.vstack(flow).copy(), \
               gt_patch.copy(), \


class DataSetV_Multi(data.Dataset):
    def __init__(self, input_dir, lr_dir, gt_dir, frames=5, cropSize=32, sigma=2, gamma=2.2):
        super(DataSetV_Multi, self).__init__()
        self.input_dir = input_dir
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.cropSize = cropSize
        self.sigma = sigma
        self.gamma = gamma
        self.frames = frames

        self.names = [x for x in sorted(os.listdir(self.gt_dir)) if is_image_file(x)]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        #print(index)
        # numpy
        # scale = random.randint(10, 10)/10
        scale = random.choice([0.5, 0.75, 1])
        #scale = 1
        #interval = random.choice([-3, -2, -1, 1, 2, 3])
        interval = 1
        if self.frames == 5:
            self.inputlists = np.clip([index-interval*2, index-interval, index, index+interval, index+interval*2],0,99)
        if self.frames == 7:
            self.inputlists = np.clip([index-interval*3, index-interval*2, index-interval, index, index+interval, index+interval*2, index+interval*3],0,99)
        self.fineSize = self.cropSize//scale

        ################# For LR_Blur Images #################
        input = []
        lr_s1 = []
        lr_s2 = []
        lr_s4 = []
        input_imgs = [[Image.open(join(self.input_dir, self.names[i])), Image.open(join(self.lr_dir, self.names[i]))]\
                      for i in self.inputlists]
        left_top_w = random.randint(0, input_imgs[0][0].size[0] - self.fineSize - 1)
        left_top_h = random.randint(0, input_imgs[0][0].size[1] - self.fineSize - 1)
        isFilp = random.randint(0, 1)
        rotation_times = random.randint(0, 3)
        for input_img, lr_img in input_imgs:
            input_patch = input_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
            lr_patch = lr_img.crop(
            (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))

            # input_patch.resize(face_size, Image.ANTIALIAS)
            input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            lr_patch_s1 = lr_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            lr_patch_s2 = lr_patch.resize((self.cropSize // 2, self.cropSize // 2), Image.BICUBIC)
            lr_patch_s4 = lr_patch.resize((self.cropSize // 4, self.cropSize // 4), Image.BICUBIC)


            input_patch = np.array(input_patch, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            lr_patch_s1 = np.array(lr_patch_s1, dtype=np.float32) / 255
            lr_patch_s1 = lr_patch_s1.transpose((2, 0, 1))
            lr_patch_s2 = np.array(lr_patch_s2, dtype=np.float32) / 255
            lr_patch_s2 = lr_patch_s2.transpose((2, 0, 1))
            lr_patch_s4 = np.array(lr_patch_s4, dtype=np.float32) / 255
            lr_patch_s4 = lr_patch_s4.transpose((2, 0, 1))
            # randomly flip
            if isFilp == 0:
                input_patch  = np.flip(input_patch, 2)
                lr_patch_s1 = np.flip(lr_patch_s1, 2)
                lr_patch_s2 = np.flip(lr_patch_s2, 2)
                lr_patch_s4 = np.flip(lr_patch_s4, 2)
            # randomly rotation
            input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
            input.append(input_patch)
            lr_patch_s1 = np.rot90(lr_patch_s1, rotation_times, (1, 2))
            lr_s1.append(lr_patch_s1)
            lr_patch_s2 = np.rot90(lr_patch_s2, rotation_times, (1, 2))
            lr_s2.append(lr_patch_s2)
            lr_patch_s4 = np.rot90(lr_patch_s4, rotation_times, (1, 2))
            lr_s4.append(lr_patch_s4)

        ################# For GT and LR_Deblur Images #################
        gt_img = Image.open(os.path.join(self.gt_dir, self.names[index])).convert('RGB')
        input_img = Image.open(os.path.join(self.input_dir, self.names[index])).convert('RGB')
        size_in = input_img.size
        new_size_in = tuple([int(x * 4) for x in size_in])
        bicubic_img = input_img.resize(new_size_in, Image.BICUBIC)

        gt_patch = gt_img.crop(
            (left_top_w*4, left_top_h*4, left_top_w *4+ self.fineSize*4, left_top_h*4 + self.fineSize*4))
        bicubic_patch = bicubic_img.crop(
            (left_top_w*4, left_top_h*4, left_top_w*4 + self.fineSize*4, left_top_h*4 + self.fineSize*4))
        gt_patch = gt_patch.resize((self.cropSize*4, self.cropSize*4), Image.BICUBIC)
        bicubic_patch = bicubic_patch.resize((self.cropSize * 4, self.cropSize * 4), Image.BICUBIC)

        gt_patch = np.array(gt_patch, dtype=np.float32) / 255
        gt_patch = gt_patch.transpose((2, 0, 1))
        bicubic_patch = np.array(bicubic_patch, dtype=np.float32) / 255
        bicubic_patch = bicubic_patch.transpose((2, 0, 1))
        # randomly flip
        if isFilp == 0:
            gt_patch     = np.flip(gt_patch, 2)
            bicubic_patch= np.flip(bicubic_patch, 2)
        # randomly rotation
        gt_patch       = np.rot90(gt_patch, rotation_times, (1, 2))
        bicubic_patch = np.rot90(bicubic_patch, rotation_times, (1, 2))


        return np.vstack(input).copy(), \
               np.vstack(lr_s4).copy(), \
               np.vstack(lr_s2).copy(), \
               np.vstack(lr_s1).copy(),\
               gt_patch.copy(), \
               bicubic_patch.copy()