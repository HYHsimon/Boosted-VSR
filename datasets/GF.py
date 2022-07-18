
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import re
import shutil
import os.path as osp

# Parameter for main camera
############ For GF_test_xt1 ############
def parameter_adaptive_main(scale, ISO, gain_base, min_detail_th_base, max_detail_th_base, filter_radius_base, sigma_base):

    filter_radius = round(filter_radius_base * scale * 2 / 3)
    sigma = sigma_base + (scale - 1) * 0.025

    if ISO <= 500:
        gain = 1.05 * gain_base
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节

    elif ISO <= 1000:
        gain_res = 0.05 * gain_base / 500 * (ISO - 500)
        gain = 1.05 * gain_base - gain_res
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节

    elif ISO <= 2000:
        gain_res = 0.2 * gain_base / 1000 * (ISO - 1000)
        gain = 1 * gain_base - gain_res
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节

    elif ISO <= 4000:
        gain_res = 0.3 * gain_base / 2000 * (ISO - 2000)
        gain = 0.8 * gain_base - gain_res
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = (min_detail_th_base) * gain  # 小于此阈值的细节不认为是细节
    else:
        gain_res = 0.5 * gain_base / 6000 * (ISO - 4000)
        gain = 0.5 * gain_base - gain_res
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = (min_detail_th_base) * gain  # 小于此阈值的细节不认为是细节

    return filter_radius, sigma, gain, max_detail_th, min_detail_th

############ For GF_test_xm ############
# def parameter_adaptive_main(scale, ISO, gain_base, min_detail_th_base, max_detail_th_base, filter_radius_base, sigma_base):
#
#     filter_radius = round(filter_radius_base * scale * 2 / 3)
#     sigma = sigma_base + (scale - 1) * 0.025
#
#     if ISO <= 500:
#         gain = 1.3 * gain_base
#         max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
#         min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节
#
#     elif ISO <= 1000:
#         gain_res = 0.2 * gain_base / 500 * (ISO - 500)
#         gain = 1.3 * gain_base - gain_res
#         max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
#         min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节
#
#     elif ISO <= 2000:
#         gain_res = 0.3 * gain_base / 1000 * (ISO - 1000)
#         gain = 1.1 * gain_base - gain_res
#         max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
#         min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节
#
#     elif ISO <= 4000:
#         gain_res = 0.3 * gain_base / 2000 * (ISO - 2000)
#         gain = 0.8 * gain_base - gain_res
#         max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
#         min_detail_th = (min_detail_th_base) * gain  # 小于此阈值的细节不认为是细节
#     else:
#         gain_res = 0.5 * gain_base / 6000 * (ISO - 4000)
#         gain = 0.5 * gain_base - gain_res
#         max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
#         min_detail_th = (min_detail_th_base) * gain  # 小于此阈值的细节不认为是细节
#
#     return filter_radius, sigma, gain, max_detail_th, min_detail_th

# Parameter for tele-camera
def parameter_adaptive_tele(scale, ISO, gain_base, min_detail_th_base, max_detail_th_base, filter_radius_base, sigma_base, coef=1.2): #主摄等效焦距24.52 长焦等效焦距88.92

    filter_radius = round(filter_radius_base * scale * coef * 2 / 3)
    sigma = (sigma_base + (scale - 1) * 0.025) * coef

    if ISO <= 500:
        gain = 1.05 * gain_base
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节

    elif ISO <= 1000:
        gain_res = 0.05 * gain_base / 500 * (ISO - 500)
        gain = 1.05 * gain_base - gain_res
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节

    elif ISO <= 2000:
        gain_res = 0.2 * gain_base / 1000 * (ISO - 1000)
        gain = 1 * gain_base - gain_res
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = min_detail_th_base * gain  # 小于此阈值的细节不认为是细节

    elif ISO <= 4000:
        gain_res = 0.3 * gain_base / 2000 * (ISO - 2000)
        gain = 0.8 * gain_base - gain_res
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = (min_detail_th_base) * gain  # 小于此阈值的细节不认为是细节
    else:
        gain_res = 0.5 * gain_base / 6000 * (ISO - 4000)
        gain = 0.5 * gain_base - gain_res
        max_detail_th = max_detail_th_base * gain  # 加到原图上的细节的最大值，阈值越大，越容易发白
        min_detail_th = (min_detail_th_base) * gain  # 小于此阈值的细节不认为是细节

    return filter_radius, sigma, gain, max_detail_th, min_detail_th

#Read XML file
def XML_Parsing(xml_path):
    root = ET.parse(xml_path).getroot()
    # print("root type:", type(root))
    for meta_info in root.findall('meta_info'):
        ISO = meta_info.find('ISO').text
        scale = meta_info.find('scale').text
        print('ISO is: {}'.format(ISO))
        print('scale is: {}'.format(scale))
    return float(ISO), float(scale)

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def edge_suppress(yuv, enhanced_detail):
    # Suppress white
    # mask_w = yuv[0].copy()
    # mask_w[mask_w > 120] = -4
    # mask_w[mask_w > 170] = -2
    # mask_w[mask_w > 220 ] = -1
    # mask_w[mask_w > 0] = 0
    # mask_w = mask_w / 10
    # mask0 = enhanced_detail * mask_w
    # mask0[mask0  >= 0] = 0
    # mask0 [mask0  < 0] = -1
    # mask_w = mask_w * mask0
    # mask[mask == 0] = -1
    # enhanced_detail = enhanced_detail * (-mask)

    # Suppress black
    mask = yuv[0].copy()
    mask[mask < 80] = -4
    # mask[mask < 60 ] = -4
    mask[mask < 40] = -2
    # mask[mask < 180] = -6
    mask[mask > 0] = 0
    mask = mask / -10
    mask1 = enhanced_detail * mask
    mask1[mask1 >= 0] = 0
    mask1[mask1 < 0] = 1
    mask = mask * mask1
    mask[mask == 0] = 1
    enhanced_detail = enhanced_detail * mask

    return enhanced_detail

def adaptive_guided_filter(input_folder,
                           gain_base=0.6, min_detail_th_base=2, max_detail_th_base=30, uv_smooth=True, iso_th=1600, filter_radius_base=3, sigma_base=0.1, suppress=False, sensor_type='MAIN', scale_default=3):
    # 文件格式，"数字-" 开头，如58-xxx.jpg, 若数字后面不是"-"，需要改下output_file_path
    output_folder = '/opt/tiger/BasicVSR/results/IconVSR_Youku_clip_BD7_re/Epoch31/xigua_gf/00002_480p.mp4'
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
    input_folder_list = sorted_aphanumeric(os.listdir(input_folder))
    # 增强文件所包含的名字
    image_include_name = ".png"

    num = 0
    for img_path in input_folder_list:
        if image_include_name in img_path.lower():
            image_path = input_folder + "/" + img_path
            img_index = re.findall('\d+', img_path)[0]
            # if int(img_index) < 143:
            #     continue

            image = cv2.imread(image_path).astype(np.float32)
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv = cv2.split(yuv)


            ISO = 50
            scale = 3

            # 自适应增强参数，值越大增强越大, 调sigma不增加计算复杂度，增大filter_radius也会增加计算复杂度, 举例：sigma/fr/gain=0.1/6/1.0/30
            ############
            ISO = int(ISO)
            if sensor_type == 'MAIN':
                filter_radius, sigma, gain, max_detail_th, min_detail_th = parameter_adaptive_main(scale, ISO,
                                                                                                   gain_base,
                                                                                                   min_detail_th_base,
                                                                                                   max_detail_th_base,
                                                                                                   filter_radius_base,
                                                                                                   sigma_base)
            elif sensor_type == 'TELE':
                filter_radius, sigma, gain, max_detail_th, min_detail_th = parameter_adaptive_tele(scale, ISO,
                                                                                                   gain_base,
                                                                                                   min_detail_th_base,
                                                                                                   max_detail_th_base,
                                                                                                   filter_radius_base,                                                                                                 sigma_base)
            if ISO < 10000:
                y_lp = cv2.ximgproc.guidedFilter(yuv[0], yuv[0], filter_radius, pow(sigma*255, 2))  # opencv和paper中sigma设置不太一样
            else:
                y_lp = yuv[0]

            if uv_smooth is True:
                u_lp = cv2.ximgproc.guidedFilter(yuv[0], yuv[1], filter_radius, pow(sigma * 255, 2))  # opencv和paper中sigma设置不太一样
                v_lp = cv2.ximgproc.guidedFilter(yuv[0], yuv[2], filter_radius, pow(sigma * 255, 2))  # opencv和paper中sigma设置不太一样
                yuv[1] = u_lp
                yuv[2] = v_lp

            # 增强y
            y_detail = yuv[0] - y_lp

            enhanced_detail = gain*y_detail
            enhanced_detail[enhanced_detail > max_detail_th] = max_detail_th
            enhanced_detail[enhanced_detail < -max_detail_th] = -max_detail_th
            enhanced_detail[abs(enhanced_detail) < min_detail_th] = 0

            if suppress == True:
                enhanced_detail = edge_suppress(yuv, enhanced_detail)

            yuv[0] = yuv[0] + enhanced_detail

            # Output
            out_image = cv2.merge(yuv)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_YUV2BGR)

            # output_file_path = output_folder + img_index + "_gf_sigma%.2f"%sigma + "_fr%d"%filter_radius + "_gain%.2f"%gain + "_Max-MinDetailTh=%.2f"%max_detail_th + "-%.2f"%min_detail_th + "_Suppress=" + str(suppress) + "." + img_path.split(".")[-1]
            output_file_path = output_folder + "/" + img_path.split(".")[0] + "_gf" + '.' + img_path.split(".")[-1]

            print("output enhanced file: {} completed!".format(output_file_path))
            cv2.imwrite(output_file_path, out_image)

            # break

            num += 1
    print(num)
    pass

# C:/Users/Admin/Desktop/Deblur0706/night
# 'D:/PytorchProjects/mmsr_arch68/results/Arch68Y_0704/Sim-HR'
adaptive_guided_filter(input_folder='/opt/tiger/BasicVSR/results/IconVSR_Youku_clip_BD7_re/Epoch31/xigua/00002_480p.mp4',
                       suppress=False)