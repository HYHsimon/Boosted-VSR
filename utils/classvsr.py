import numpy as np
import torch
from visualization import show_tensor_image

def crop_cpu(img, crop_sz, step):

    h, w, c = img.size()

    left_H = h // step * step + crop_sz - h
    left_W = w // step * step + crop_sz - w

    img = torch.cat([img, torch.flip(img, [0])], 0)[:h + left_H, :]
    img = torch.cat([img, torch.flip(img, [1])], 1)[:, :w + left_W]

    # if left_H :
    #     left_H = torch.zeros(left_H, w, c).to(opt.device)
    #     img = torch.cat((img, left_H), dim=0)
    #
    # if left_W:
    #     left_W = torch.zeros(h, left_W, c).to(opt.device)
    #     img = torch.cat((img, left_W), dim=1)

    # show_tensor_image(img.permute(2, 0, 1))

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            lr_list.append(crop_img)
    h = x + crop_sz
    w = y + crop_sz
    return lr_list, num_h, num_w, h, w

def combine(sr_list,num_h, num_w,h,w,patch_size,step, scale):
    index=0
    sr_img = sr_list[index].new_zeros(h*scale, w*scale, sr_list[index].shape[2])
    for i in range(num_h):
        for j in range(num_w):
            sr_img[i*step*scale:i*step*scale+patch_size*scale, j*step*scale:j*step*scale+patch_size*scale, :]+=sr_list[index]
            index+=1
    # sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:,j*step*scale:j*step*scale+(patch_size-step)*scale,:]/=2

    for i in range(1,num_h):
        sr_img[i*step*scale:i*step*scale+(patch_size-step)*scale,:,:]/=2
    return sr_img

def combine2(sr_list,num_h, num_w,h,w,patch_size,step, scale):
    index=0
    sr_img = sr_list[index].new_zeros(h*scale, w*scale, sr_list[index].shape[2])
    for i in range(num_h):
        for j in range(num_w):

            for (m, x) in zip(range(h), range(i*step*scale, i*step*scale+patch_size*scale)):
                for (n, y) in zip(range(w), range(j*step*scale, j*step*scale+patch_size*scale)):

                    if torch.mean(sr_img[x, y, :]) == 0 :

                        sr_img[x, y, :] += sr_list[index][m, n, :]

                    else:

                        sr_img[x, y, :] += sr_list[index][m, n, :]
                        sr_img[x, y, :] /= 2

            index+=1

    return sr_img

def crop_border(v, crop_border):

    return v[:, :, :, crop_border:-crop_border, crop_border:-crop_border]
