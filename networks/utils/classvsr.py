import numpy as np
import torch

def match(img):
    _, _, _, h, w= img.size()

    left_H = 0
    left_W = 0

    if h % 4 :
        left_H = (h // 4 + 1) * 4 - h
    if w % 4:
        left_W = (w // 4 + 1) * 4 - w

    img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :h + left_H, :]
    img = torch.cat([img, torch.flip(img, [4])], 4)[:, :, :, :, :w + left_W]

    return img

def crop_cpu(img, crop_sz, step, device):
    h, w, c = img.size()

    left_H = h // step * step + crop_sz - h
    left_W = w // step * step + crop_sz - w

    img = torch.cat([img, torch.flip(img, [0])], 0)[:h + left_H, :]
    img = torch.cat([img, torch.flip(img, [1])], 1)[:, :w + left_W]

    # show_tensor_image(img.permute(2, 0, 1))

    # if left_H:
    #     left_H = torch.zeros(left_H, w, c).to(device)
    #     img = torch.cat((img, left_H), dim=0)
    #
    # if left_W:
    #     left_W = torch.zeros(h, left_W, c).to(device)
    #     img = torch.cat((img, left_W), dim=1)

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
    sr_img = sr_list[index].new_zeros(h*scale, w*scale, 3)
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


def crop_border(v, crop_border):

    return v[:, :, :, crop_border:-crop_border, crop_border:-crop_border]


import matplotlib.pyplot as plt
def show_tensor_image(tensor, is_Variable=True):
# input 3D or 4D Tensor (C*H*W or N*C*H*W)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if is_Variable:
        img = tensor.cpu().data[0].numpy().transpose((1,2,0))
    else:
        img = tensor[0].numpy().transpose((1,2,0))

    plt.imshow(img)
    plt.axis('off')
    plt.show()
