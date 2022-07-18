import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torchvision
from PIL import Image
import cv2

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


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

def save_tensor_image(tensor, path):
# input 4D Varialble or Tensor (N*C*H*W)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    try:
        img = tensor[0].numpy().transpose((1,2,0))
    except:
        img = tensor[0].cpu().numpy().transpose((1,2,0))        
    img = np.uint8(img*255)
    Image.fromarray(img).save(path)

############################################################################

def show_label1(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()

def show_seg(gt, pred, sr, label_heatmap, image_cls):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig1, axes1 = plt.subplots(1, 3)
    ax1, ax2, ax3 = axes1

    fig2, axes2 = plt.subplots(2, 1)
    ax4, ax5 = axes2

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    axes1[0].axis('off')
    ax1.imshow(gt)

    ax2.set_title('pred')
    axes1[1].axis('off')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    ax3.set_title('sr')
    axes1[2].axis('off')
    ax3.imshow(sr)

    ax4.imshow(label_heatmap)
    axes2[0].axis('off')
    ax5.imshow(image_cls)
    axes2[1].axis('off')

    plt.show()


def show_features_test(features, batch, num_chns):

    size = features.data[0].size()
    #feature_maps = features.cpu().data[batch].view(size[0]//num_chns, num_chns, size[1], size[2])
    feature_maps = features.cpu().data[batch,15*32:16*32].view(32, num_chns, size[1], size[2])
    image_features = torchvision.utils.make_grid(feature_maps)
    image_features = image_features.numpy().transpose((1,2,0))

    #fig, ax = plt.subplots(1, 1)
    #ax.imshow(image_features)
    plt.figure()
    plt.imshow(image_features)
    plt.show()


def show_ssloss_seg(gt, pred_gt, sr, pred_sr):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig1, axes1 = plt.subplots(2, 2)
    ax1_1, ax1_2, ax2_1, ax2_2= axes1


    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1_1.set_title('gt')
    axes1[0].axis('off')
    ax1_1.imshow(gt)

    ax1_2.set_title('pred_gt')
    axes1[1].axis('off')
    ax1_2.imshow(pred_gt, cmap=cmap, norm=norm)

    ax2_1.set_title('sr')
    axes1[2].axis('off')
    ax2_1.imshow(sr)

    ax2_2.set_title('pred_sr')
    axes1[3].axis('off')
    ax2_2.imshow(pred_sr, cmap=cmap, norm=norm)

    plt.show()

def show_ssloss_seg1(hr, gt_label, hr_label, sr_label):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig1, axes1 = plt.subplots(2, 2)
    ax1_1, ax1_2 = axes1[0]
    ax2_1, ax2_2 = axes1[1]


    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1_1.set_title('gt')
    axes1[0,0].axis('off')
    ax1_1.imshow(hr)

    ax1_2.set_title('gt_label')
    axes1[0,1].axis('off')
    ax1_2.imshow(gt_label, cmap=cmap, norm=norm)

    ax2_1.set_title('hr_label')
    axes1[1,0].axis('off')
    ax2_1.imshow(hr_label, cmap=cmap, norm=norm)

    ax2_2.set_title('sr_label')
    axes1[1,1].axis('off')
    ax2_2.imshow(sr_label, cmap=cmap, norm=norm)

    plt.show()


def show_label(label, is_label=True):
# input 3D label (N*H*W) Tensor or 4D seg Tensor (N*C*H*W)
    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5),(1,1,1)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,255]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    if is_label:
        label = np.asarray(label[0].numpy(), dtype=np.int)
    else:
        label = label[0].numpy()
        label = label.transpose(1, 2, 0)
        label = np.asarray(np.argmax(label, axis=2), dtype=np.int)


    plt.imshow(label, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.show()


def show_features(features, batch, num_chns):
    # input 4D Varialble (N*C*H*W)
    size = features.cpu().data[0].size()
    feature_maps = features.cpu().data[batch].view(size[0]//num_chns, num_chns, size[1], size[2])
    image_features = torchvision.utils.make_grid(feature_maps, padding=3, pad_value=1)
    image_features = image_features.numpy().transpose((1,2,0))

    #fig, ax = plt.subplots(1, 1)
    #ax.imshow(image_features)
    plt.imshow(image_features, interpolation='nearest')
    plt.axis('off')
    plt.show()

def show_features_enhanced(features, batch, num_chns, normalize=True, pseudo=True, img_index=0, ft_index=0):
    # first crop the features
    #features = features[:,:,6:166,44:204]
    # input 4D Tensor (N*C*H*W)
    size = features[0].size()
    feature_maps = features[batch].view(size[0]//num_chns, num_chns, size[1], size[2])
    image_features = torchvision.utils.make_grid(feature_maps.cpu(), nrow=16, padding=3, pad_value=1, normalize=normalize)
    image_features = image_features.numpy().transpose((1,2,0))

    if pseudo:
        image_features = image_features[:, :, 0]
        imgplot = plt.imshow(image_features)
        imgplot.set_cmap('nipy_spectral')
        #plt.colorbar()
    else:
        plt.imshow(image_features)
    plt.axis('off')
    plt.show()

    fname = 'C:\Datasets\GOPRO_Large\Validation_ft_4x\Results\%d_%d' % (img_index, ft_index)
    plt.imsave(fname, image_features, cmap='nipy_spectral')

def show_input(lr, lr_label, hr, hr_label):
    # input 4D Varialble images and 3D label (N*H*W) Tensor
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig1, axes1 = plt.subplots(2, 2)
    ax1_1, ax1_2 = axes1[0]
    ax2_1, ax2_2 = axes1[1]


    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5),(1,1,1)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,255]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    lr_img = lr.cpu().data[0].numpy().transpose((1, 2, 0))
    hr_img = hr.cpu().data[0].numpy().transpose((1, 2, 0))

    lr_gt = np.asarray(lr_label[0].numpy(), dtype=np.int)
    hr_gt = np.asarray(hr_label[0].numpy(), dtype=np.int)

    ax1_1.set_title('lr')
    axes1[0,0].axis('off')
    ax1_1.imshow(lr_img, interpolation='bilinear')

    ax1_2.set_title('lr_label')
    axes1[0,1].axis('off')
    ax1_2.imshow(lr_gt, cmap=cmap, norm=norm)

    ax2_1.set_title('hr')
    axes1[1,0].axis('off')
    ax2_1.imshow(hr_img, interpolation='bilinear')

    ax2_2.set_title('hr_label')
    axes1[1,1].axis('off')
    ax2_2.imshow(hr_gt, cmap=cmap, norm=norm)

    plt.show()


def show_output(sr, sr_blur, sr_edge, hr, hr_blur, hr_edge):
    # input 4D Varialble images
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig1, axes1 = plt.subplots(2, 3)
    ax1_1, ax1_2, ax1_3 = axes1[0]
    ax2_1, ax2_2, ax2_3 = axes1[1]

    sr_img = sr.cpu().data[0].numpy().transpose((1, 2, 0))
    sr_blur_img = sr_blur.cpu().data[0].numpy().transpose((1, 2, 0))
    size = sr_edge[0].size()
    feature_maps = sr_edge[0].view(size[0]//3, 3, size[1], size[2])
    image_features1 = torchvision.utils.make_grid(feature_maps, padding=3, pad_value=1, normalize=False)
    image_features1 = image_features1.numpy().transpose((1,2,0))
    image_features1 = image_features1[:, :, 0]


    hr_img = hr.cpu().data[0].numpy().transpose((1, 2, 0))
    hr_blur_img = hr_blur.cpu().data[0].numpy().transpose((1, 2, 0))
    sr_img = sr.cpu().data[0].numpy().transpose((1, 2, 0))
    sr_blur_img = sr_blur.cpu().data[0].numpy().transpose((1, 2, 0))
    size = hr_edge[0].size()
    feature_maps = hr_edge[0].view(size[0]//3, 3, size[1], size[2])
    image_features2 = torchvision.utils.make_grid(feature_maps, padding=3, pad_value=1, normalize=False)
    image_features2 = image_features2.numpy().transpose((1,2,0))
    image_features2 = image_features2[:, :, 0]


    ax1_1.set_title('sr')
    axes1[0,0].axis('off')
    ax1_1.imshow(sr_img)

    ax1_2.set_title('sr_blur')
    axes1[0,1].axis('off')
    ax1_2.imshow(sr_blur_img)

    ax1_3.set_title('sr_edge')
    axes1[0,2].axis('off')
    imgplot1= ax1_3.imshow(image_features1)
    imgplot1.set_cmap('nipy_spectral')
    #plt.colorbar()

    ax2_1.set_title('hr')
    axes1[1,0].axis('off')
    ax2_1.imshow(hr_img)

    ax2_2.set_title('hr_blur')
    axes1[1, 1].axis('off')
    ax2_2.imshow(hr_blur_img)

    ax2_3.set_title('hr_edge')
    axes1[1, 2].axis('off')
    imgplot2= ax2_3.imshow(image_features2)
    imgplot2.set_cmap('nipy_spectral')
    #plt.colorbar()

    plt.show()

# show flow
def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[0, 0]
    v = flow[0, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # maxu = max(maxu, np.max(u))
    # minu = min(minu, np.min(u))
    #
    # maxv = max(maxv, np.max(v))
    # minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def show_flow(flow,path):
    flow_numpy = flow.cpu().numpy()
    img = flow_to_image(flow_numpy)
    # plt.imshow(img)
    # plt.show()

    img = np.uint8(img * 255)
    Image.fromarray(img).save(path)