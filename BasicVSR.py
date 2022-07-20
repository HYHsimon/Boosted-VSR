from __future__ import print_function
import os
from os.path import join
import torch
import time
import numpy as np
from math import log10
from visualization import save_tensor_image

def test(testloader, model, criterion, opt):
    psnrs_folder = []

    with torch.no_grad():
        for iteration, batch in enumerate(testloader, 1):
            # input, targetdeblur, targetsr
            if not opt.isReal:
                LQ_all = batch[0].to(opt.device)
                GT_all = batch[1].to(opt.device)
                names = batch[2]
                dir_name = batch[3]
            else:
                LQ_all = batch[0].to(opt.device)
                names = batch[1]
                dir_name = batch[2]

            # Backward
            start_time = time.perf_counter()  # -------------------------begin to deal with the video

            n, t, c, h, w = LQ_all.size()

            SR_ref = model(LQ_all)

            ################### save image ###################
            if opt.isSave:
                SR_ref_clipped = torch.clamp(SR_ref, 0, 1)

                video_name = dir_name[0].split('/')[-1]  # name for test video

                root = join(opt.save_path, video_name)

                if not os.path.exists(root):
                    os.makedirs(root)

                for i in range(t):
                    frame_index = iteration - 1 + i
                    frame_name = names[frame_index][0]
                    path = os.path.join(root, frame_name)
                    save_tensor_image(SR_ref_clipped[:, i], path)

            for i in range(t):
                ################### calculate psnr ###################
                if not opt.isReal:
                    mse = criterion(SR_ref[:,i], GT_all[:,i])
                    if mse == float("inf"):
                        mse = 1

                    psnr = 10 * log10(1 / mse)
                    psnrs_folder.append(psnr)
                    opt.psnrs_all.append(psnr)
                else:
                    psnrs_folder.append(0)
                    opt.psnrs_all.append(0)

        torch.cuda.synchronize()  # wait for CPU & GPU time syn
        evalation_time = time.perf_counter() - start_time  # ---------finish all the videos
        print("Folder Avg. PSNR:{:4f} Time: {:4f}".format(np.mean(psnrs_folder), evalation_time / t))
        return np.mean(psnrs_folder)


def model_test(testloader, model, opt):
    model = model.to(opt.device)
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion = criterion.to(opt.device)
    psnr = test(testloader, model, criterion, opt)
    return psnr

