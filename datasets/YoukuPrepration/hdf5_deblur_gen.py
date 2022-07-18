import h5py as h5
import os
import cv2
from PIL import Image
import numpy as np
import random

def img_sharp(image):
   kernel = np.array([[0, -0.4, 0], [-0.4, 2.6, -0.4], [0, -0.4, 0]], np.float32)
   dst = cv2.filter2D(image, -1, kernel=kernel)
   return dst

def img_blur(image):
    sigma = random.uniform(0.3, 0.7)
    dst = cv2.GaussianBlur(image, (3,3), sigma)
    return dst

class H5Dataset(object):
    def __init__(self, h5py_path, hq_folder, lq_folder, hq_shape, lq_shape, scale, type,
                 chunks=True, tranform=None, image_filter=None):
        assert not os.path.exists(h5py_path), 'h5py_path is exsiting'
        self.h5py_path = h5py_path
        self.hq_folder = hq_folder
        self.lq_folder = lq_folder
        self.hq_shape = hq_shape
        self.lq_shape = lq_shape
        self.scale = scale
        self.type = type

        self.chunks = chunks
        self.tranform = tranform if tranform is not None else self.default_tranform
        self.image_filter = image_filter if image_filter is not None else self.default_filter

        self.h5file = self.create_h5py(self.h5py_path)
        self.hq_dset = self.create_dataset("hq", (256,) + hq_shape, (None,) + hq_shape, chunks, self.type)
        self.lq_dset = self.create_dataset("lq", (256,) + lq_shape, (None,) + lq_shape, chunks, self.type)
        self.count = 0

    def default_filter(self, input):
        return input

    def default_tranform(self, hq_image, lq_image):
        # assert hq_image.shape == lq_image.shape
        hq_image = hq_image.astype(np.float32) / 255.
        # hq_image1 = hq_image.copy()
        lq_image = lq_image.astype(np.float32) / 255.


        # hq_image_Yuv = util.channel_convert(hq_image.shape[2], 'Y', [hq_image])[0]
        # hq_image_Y = util.channel_convert(hq_image1.shape[2], 'y', [hq_image1])[0]
        if np.ndim(hq_image) == 3:
            hq_image_Yuv = cv2.cvtColor(hq_image, cv2.COLOR_BGR2YUV)
            lq_image_Yuv = cv2.cvtColor(lq_image, cv2.COLOR_BGR2YUV)
            hq_image_Y = hq_image_Yuv[:,:,0:1]
            lq_image_Y = lq_image_Yuv[:,:,0:1]
        elif np.ndim(hq_image) == 2:
            hq_image_Y = np.expand_dims(hq_image, axis=2)
            lq_image_Y = np.expand_dims(lq_image, axis=2)
        # hq_image_Y = (hq_image_Y * 255).astype(np.uint8)
        # lq_image_Y = (lq_image_Y * 255).astype(np.uint8)

        # hq_image_Y = img_sharp(hq_image_Y)
        # hq_image_Y = np.expand_dims(hq_image_Y, axis=2)
        # lq_image_Y = img_blur(lq_image_Y)
        # noise_idx = random.randint(0, 50)
        # noise_scale = 0.02 * noise_idx / 50
        noise_scale = 0.01
        lq_image_Y = np.clip(lq_image_Y + np.random.randn(*lq_image_Y.shape) * noise_scale, 0, 1)
        # lq_image_Y = np.expand_dims(lq_image_Y, axis=2)
        H, W, _ = lq_image_Y.shape
        lq_image_Y = ((lq_image_Y*255).round()).astype(np.uint8)
        hq_image_Y = ((hq_image_Y * 255).round()).astype(np.uint8)
        hq_patches = []
        lq_patches = []
        # Image.fromarray(lq_image_Y[:, :, 0]).show()

        stride = self.lq_shape[0] //2
        h_nums = (H - self.lq_shape[0] - 1) // stride + 1
        w_nums = (W - self.lq_shape[1] - 1) // stride + 1
        # h_nums = H // self.lq_shape[0]
        # w_nums = W // self.lq_shape[1]
        for h in range(h_nums):
            for w in range(w_nums):
                rnd_h = h*stride
                rnd_w = w*stride
                # rnd_h = h*self.lq_shape[0]
                # rnd_w = w*self.lq_shape[1]
                lq_patch = lq_image_Y[rnd_h:rnd_h + self.lq_shape[0], rnd_w:rnd_w + self.lq_shape[1], :]
                rnd_h_GT, rnd_w_GT = int(rnd_h*self.scale), int(rnd_w*self.scale)
                hq_patch = hq_image_Y[rnd_h_GT:rnd_h_GT + self.hq_shape[0], rnd_w_GT:rnd_w_GT + self.hq_shape[1], :]
                # hq_patch = hq_patch.transpose(2, 0, 1)
                # lq_patch = lq_patch.transpose(2, 0, 1)
                hq_patches.append(hq_patch)
                lq_patches.append(lq_patch)

        return hq_patches, lq_patches

    def create_h5py(self, h5py_path=None):
        if h5py_path is None:
            h5py_path = self.h5py_path
        dirname, filename = os.path.split(h5py_path)
        if dirname != "" and not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except:
                pass
        self.h5file = h5.File(self.h5py_path, 'w')
        return self.h5file

    def create_dataset(self, tag, shape, maxshape, chunks, type=np.float32):
        dset = self.h5file.create_dataset(tag, shape=shape, maxshape=maxshape, chunks=chunks, dtype=type)
        return dset

    def center_crop(self, image, format="CHW"):
        _, H, W = self.dataset_shape
        if format == "CHW":
            c, h, w = image.shape
            left = (h - H) // 2
            top = (w - W) // 2
            image = image[:, left:left + H, top:top + W]
        elif format == "HWC":
            h, w, c = image.shape
            left = (h - H) // 2
            top = (w - W) // 2
            image = image[left:left + H, top:top + W, :]
        else:
            raise NotImplementedError("not support format")
        return image

    def write2dset(self, hq_frame=None, lq_frame=None):
        """ todo
        :param hq_frame:
        :param lq_frame:
        :return:
        """
        if len(self.hq_dset) <= self.count:
            self.hq_dset.resize((len(self.hq_dset) + 256,) + self.hq_shape)
            self.lq_dset.resize((len(self.lq_dset) + 256,) + self.lq_shape)
        assert hq_frame.shape == self.hq_shape and lq_frame.shape == self.lq_shape
        self.hq_dset[self.count] = hq_frame
        self.lq_dset[self.count] = lq_frame
        self.count += 1
        # print(self.count)

    def remove_empty(self):
        self.hq_dset.resize((self.count,) + self.hq_shape)
        self.lq_dset.resize((self.count,) + self.lq_shape)

    def write_folder(self, hq_folder=None, lq_folder=None):
        if hq_folder is None:
            hq_folder = self.hq_folder
        if lq_folder is None:
            lq_folder = self.lq_folder

        for hq_image, lq_image in zip(sorted(os.listdir(hq_folder)), sorted(os.listdir(lq_folder))):
            # self.count+=1
            # continue
            print(hq_image)
            hq_image_path = os.path.join(hq_folder, hq_image)
            lq_image_path = os.path.join(lq_folder, lq_image)
            hq_frame = cv2.imread(hq_image_path, cv2.IMREAD_UNCHANGED)
            lq_frame = cv2.imread(lq_image_path, cv2.IMREAD_UNCHANGED)
            hq_patches, lq_patches = self.tranform(hq_frame, lq_frame)
            nums = len(hq_patches)
            results = random.sample(range(nums), nums)
            for i in results:
                self.write2dset(hq_patches[i], lq_patches[i])
        self.remove_empty()



h5_creat = H5Dataset(h5py_path= 'D:\\Datasets\\Simulations\\Deblur\\HDF5\\train_Deblur_focus-v3_stride.h5', hq_folder='D:\\Datasets\\Simulations\\Deblur\\Train\\HR_focus_0628_sub', lq_folder='D:\\Datasets\\Simulations\\Deblur\\Train\\HR_sub', \
                      scale=1, hq_shape=(128,128,1), lq_shape=(128,128,1), type=np.uint8)

h5_creat.write_folder()

