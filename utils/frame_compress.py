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
from PIL import Image
import random

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

VIDEO_EXTENSIONS = ['.mp4', '.MP4', '.mpg', '.mov', '.MPG', '.flv', '.FLV', '.avi', '.AVI', '.y4m']

def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


frames_root = '/opt/tiger/Datasets/VSR/Youku/frames_clip/LR/BD3'
save_dir = '/opt/tiger/Datasets/VSR/Youku/frames_clip/LR/BD3_jpg'

videos_list = sorted(os.listdir(frames_root))


for video_name in videos_list:
    print(video_name)
    save_folder = os.path.join(save_dir, video_name) 
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    frames_list = sorted(os.listdir(os.path.join(frames_root, video_name)))
    for frame_name in frames_list:
        if is_image_file(frame_name):
            img = Image.open(os.path.join(frames_root, video_name, frame_name))
            save_path = os.path.join(save_folder, frame_name[:-4]+'.jpg')
            q = random.randint(45,80)
            img.save(save_path, quality=q)

