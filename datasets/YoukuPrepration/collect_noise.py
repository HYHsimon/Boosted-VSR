from PIL import Image
import numpy as np
import os.path as osp
import glob
import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--artifacts', default='tdsr', type=str, help='selecting different artifacts type')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
with open('/opt/tiger/BasicVSR/datasets/YoukuPrepration/paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)



def noise_patch(rgb_img, sp, max_var, min_mean, max_mean):
    img = rgb_img.convert('L')
    rgb_img = np.array(rgb_img)
    img = np.array(img)

    w, h = img.shape
    collect_patchs = []
    collect_noise = []
    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            patch = img[i:i + sp, j:j + sp]
            var_global = np.var(patch)
            mean_global = np.mean(patch)
            if var_global < max_var and mean_global > min_mean and mean_global < max_mean:
                rgb_patch = rgb_img[i:i + sp, j:j + sp, :]


                collect_patchs.append(rgb_patch)



    return collect_patchs


if __name__ == '__main__':

    if opt.dataset == 'df2k':
        img_dir = PATHS[opt.dataset][opt.artifacts]['source']
        print(img_dir)
        noise_dir = '/opt/tiger/Datasets/VSR/Youku/Noise/NoiseBD3_jpg_96'
        sp = 96
        max_var = 20
        min_mean = 20
        max_mean = 200
    else:
        img_dir = PATHS[opt.dataset][opt.artifacts]['hr']['train']
        noise_dir = '/home/yzt/project/Real-SR/datasets/noise1/'
        sp = 256
        max_var = 20
        min_mean = 50

    if not os.path.exists(noise_dir):
        os.mkdir(noise_dir)

    folders = sorted(os.listdir(img_dir))
    cnt = 0
    for folder in folders:
        img_paths = sorted(glob.glob(osp.join(img_dir, folder, '*.jpg')))
        for path in img_paths:
            img_name = osp.splitext(osp.basename(path))[0]
            print('**********', img_name, '**********')
            img = Image.open(path).convert('RGB')
            patchs = noise_patch(img, sp, max_var, min_mean, max_mean)
            for idx,patch in enumerate(patchs):
                save_path = osp.join(noise_dir,  '{}_{}_{:03}.jpg'.format(folder, img_name, idx))
                save_noise = osp.join(noise_dir, '{:08}.png'.format(cnt))
                cnt += 1
                print('collect:', cnt, save_path)

                Image.fromarray(patch).save(save_noise)

