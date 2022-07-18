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
from PIL import Image, ImageOps, ImageFilter


VIDEO_EXTENSIONS = ['.mp4', '.MP4', '.mpg', '.mov', '.MPG', '.flv', '.FLV', '.avi', '.AVI', '.y4m']

def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


frames_root = '/opt/tiger/BasicVSR/results/IconVSR_Youku_clip_BD7_re/Epoch31/xigua'
save_dir = '/opt/tiger/BasicVSR/results/IconVSR_Youku_clip_BD7_re/Epoch31/xigua_shp'
# if not osp.exists(save_dir):
#     os.makedirs(save_dir)
video_list = sorted(os.listdir(frames_root))
for video_name in video_list:
    if is_video_file(video_name) and video_name not in ['00002_480p.mp4', '00003_480p.mp4']:
        print(video_name)
        png_path = os.path.join(frames_root, video_name) 
        save_path = os.path.join(save_dir, video_name)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        for frame_name in sorted(os.listdir(png_path)):
            frame_path = os.path.join(png_path, frame_name) 
            ori_img = Image.open(frame_path)
            shp_img = ori_img.filter(ImageFilter.SHARPEN)
            output_path = os.path.join(save_path, frame_name) 
            shp_img.save(output_path)



    # ffmpeg -i .\000_cmp\%08d.png -r 20 -vcodec libx264 -pix_fmt yuv420p -profile:v high444 -refs 16 -crf 0 .\000_IRN.mp4


    # ffmpeg -i .\000_IRN.mp4 -start_number 0 .\000\%08d.png



# seg_dir = '/data1/VideoHazy_v3/Test/Collect/C005_re'

# cat_png_save_dir = '/data1/VideoHazy_v3/Test/Collect/C005_video'
# cat_video_save = '/data1/VideoHazy_v3/Test/Collect/C005_video/C005.mp4'


# if not osp.exists(cat_png_save_dir):
#     os.makedirs(cat_png_save_dir)


# for i in range(1):

#     hazy_images = sorted(glob.glob(os.path.join(hazy_dir, '*.JPG')))
#     hazy_segmentations = sorted(glob.glob(os.path.join(seg_dir, '*dark.jpg')))
#     edvr_segmentations = sorted(glob.glob(os.path.join(seg_dir, '*edvr_re.JPG')))
#     dehaze_segmentations = sorted(glob.glob(os.path.join(seg_dir, '*ours.jpg')))


#     for i in range(len(hazy_images)):
#         hazy_image = cv2.imread(hazy_images[i])
#         hazy_image_resize = cv2.resize(hazy_image.copy(), dsize=(1280, 360))
#         hazy_segmentation = cv2.imread(hazy_segmentations[i])
#         hazy_segmentation_resize = cv2.resize(hazy_segmentation.copy(), dsize=(1280, 360))
#         edvr_segmentation = cv2.imread(edvr_segmentations[i])
#         edvr_segmentation_resize = cv2.resize(edvr_segmentation.copy(), dsize=(1280, 360))
#         dehaze_segmentation = cv2.imread(dehaze_segmentations[i])
#         dehaze_segmentation_resize = cv2.resize(dehaze_segmentation.copy(), dsize=(1280, 360))

#         cat_image = np.concatenate((hazy_image_resize[:,:1120], hazy_segmentation_resize[:,:1120], edvr_segmentation_resize[:,:1120], dehaze_segmentation_resize[:,:1120]), axis=0)
#         cat_save_path = osp.join(cat_png_save_dir, '%05d.jpg'%i)
#         cv2.imwrite(cat_save_path, cat_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

#     output_video_path = cat_video_save
#     cmd = 'ffmpeg -r 5' + ' -i ' + osp.join(cat_png_save_dir,  '%05d.jpg') + ' ' \
#           '-vcodec libx264 -pix_fmt yuv420p -profile:v high444 -refs 16 -crf 0 ' + output_video_path
#     os.system(cmd)

    # ffmpeg -i .\000_cmp\%08d.png -r 20 -vcodec libx264 -pix_fmt yuv420p -profile:v high444 -refs 16 -crf 0 .\000_IRN.mp4


    # ffmpeg -i .\000_IRN.mp4 -start_number 0 .\000\%08d.png