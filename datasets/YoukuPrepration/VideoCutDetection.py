# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager
import os
import glob
import re
import shutil

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

VIDEO_EXTENSIONS = ['.mp4', '.MP4', '.mpg', 'mov', '.MPG', '.flv', '.FLV', '.avi', '.AVI', 'y4m']


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)

def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


video_root = '/opt/tiger/Datasets/VSR/Youku/GT' # GT video to detect the cut
GT_frame_root = '/opt/tiger/Datasets/VSR/Youku/GT/frames' # original GT frames
LR_frame_root = '/opt/tiger/Datasets/VSR/Youku/LR/LR_compress_BD10/BD10' # original LR frames
clip_root = '/opt/tiger/Datasets/VSR/Youku/frames_clip_new' # root of the cutted frames
videos = sorted(os.listdir(video_root))
videos_LR = sorted(os.listdir(LR_frame_root))
for index, video in enumerate(videos):
    if is_video_file(video):
        print(video)
        GT_frames_path = os.path.join(GT_frame_root, video)
        LR_frames_path = os.path.join(LR_frame_root, videos_LR[index])

        GT_names = [x for x in sorted(os.listdir(GT_frames_path))]
        LR_names = [x for x in sorted(os.listdir(LR_frames_path))]
        video_path = os.path.join(video_root, video)
        scenes = find_scenes(video_path)
        clip_index = 0
        for scene in scenes:
            start_frame = scene[0].frame_num
            end_frame =  scene[1].frame_num
            if end_frame - start_frame < 12:
                continue
            GT_frames_path_new = os.path.join(clip_root, 'GT', video+'_clip{:02d}'.format(clip_index))  # folder for cutted GT frames 
            LR_frames_path_new = os.path.join(clip_root, 'LR/{}'.format(LR_frame_root.split('/')[-1]), video+'_clip{:02d}'.format(clip_index)) # folder for cutted LR frames 
            if not os.path.exists(GT_frames_path_new):
                os.makedirs(GT_frames_path_new)
            if not os.path.exists(LR_frames_path_new):
                os.makedirs(LR_frames_path_new)

            for i, frame_index in enumerate(range(start_frame, end_frame)):
                GT_frame_path = os.path.join(GT_frames_path, GT_names[frame_index])
                GT_frame_path_new = os.path.join(GT_frames_path_new, f'{i:08d}.png')  
                LR_frame_path = os.path.join(LR_frames_path, LR_names[frame_index])
                LR_frame_path_new = os.path.join(LR_frames_path_new, f'{i:08d}.png')
                # shutil.copyfile(GT_frame_path, GT_frame_path_new)
                shutil.copyfile(LR_frame_path, LR_frame_path_new)
            clip_index += 1
