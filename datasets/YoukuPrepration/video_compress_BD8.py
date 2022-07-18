###################
# CVPR2021 Submission / 2020.11.22
# Combine multiple groups of frames into one video
# -pix_fmt yuv420p used for mac OS
# ffmpeg: https://www.jianshu.com/p/63dec49dc864

import os
import numpy as np
import os.path as osp
import cv2
# import skvideo.io as skvio
import glob
import random

VIDEO_EXTENSIONS = ['.mp4', '.MP4', '.mpg', 'mov', '.MPG', '.flv', '.FLV', '.avi', '.AVI', 'y4m']


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def _get_paths_from_videos(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    videos = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_video_file(fname):
                video_path = os.path.join(dirpath, fname)
                videos.append(video_path)
    assert videos, '{:s} has no valid video file'.format(path)
    return videos


videos_dir = '/opt/tiger/Datasets/VSR/Youku/GT'
save_dir = '/opt/tiger/Datasets/VSR/Youku/LR/LR_compress_BD8'
save_tmp_dir = '/opt/tiger/Datasets/VSR/Youku/LR/LR_compress_BD8_tmp'
video_list = _get_paths_from_videos(videos_dir)



if not osp.exists(save_dir):
    os.makedirs(save_dir)
if not osp.exists(save_tmp_dir):   
    os.makedirs(save_tmp_dir)
for video in video_list:
    sigma = random.uniform(0.3, 1)
    degraded_scales = [540]
    degraded_scale = random.choice(degraded_scales)
    bitrate = random.randint(1200,2000)

    output_path = os.path.join(save_dir, video.split('/')[-1][:-4] + '.mp4') 
    cmd_blur = "ffmpeg -y -i {} -c:v libx264 -pix_fmt yuv420p -vf \"gblur=sigma={}\" -crf 1 {}".format(video, sigma, output_path)
    os.system(cmd_blur)

    input_path = output_path
    output_path = os.path.join(save_tmp_dir, video.split('/')[-1][:-4] + '.mp4') 
    cmd_down = "ffmpeg -y -i {} -vf scale=-1:{} {}".format(input_path, degraded_scale, output_path)
    os.system(cmd_down)

    input_path = output_path
    output_path = os.path.join(save_dir, video.split('/')[-1][:-4] + '.mp4') 
    cmd_up = "ffmpeg -y -i {} -vf scale=1920:1080 {}".format(input_path, output_path)
    os.system(cmd_up)

    input_path = output_path
    output_path = os.path.join(save_tmp_dir, video.split('/')[-1][:-4] + '.mp4') 
    cmd_compress = "ffmpeg -y -i {} -c:v libx264 -b:v {}k {}".format(input_path, bitrate, output_path)
    os.system(cmd_compress)

    input_path = output_path
    output_path = os.path.join(save_dir, video.split('/')[-1][:-4] + '.mp4') 
    cmd_fin = "ffmpeg -y -i {} -vf scale=960:540 {}".format(input_path, output_path)
    os.system(cmd_fin)
    # 
    # bitrate = 800
    # # output_video_path = cat_video_save
    # # cmd_i2v = 'ffmpeg -r 5' + ' -i ' + osp.join(cat_png_save_dir,  '%05d.jpg') + ' ' \
    # #         '-vcodec libx264 -pix_fmt yuv420p -profile:v high444 -refs 16 -crf 0 ' + output_video_path

    # cmd_v2i = 'ffmpeg' + ' -i ' + video + ' ' \
    #         '-c:v libx264 ' + '-b:v {}k '.format(bitrate) + output_path

    # # cmd_v2i = 'ffmpeg' + ' -i ' + video + ' ' \
    # #         '-vf scale=960:540 ' + output_path
            
    # os.system(cmd_v2i)
    # # ffmpeg -i .\000_cmp\%08d.png -r 20 -vcodec libx264 -pix_fmt yuv420p -profile:v high444 -refs 16 -crf 0 .\000_IRN.mp4


    # # ffmpeg -i .\000_IRN.mp4 -start_number 0 .\000\%08d.png



