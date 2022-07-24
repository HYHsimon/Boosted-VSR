from __future__ import print_function
import os
from os.path import join
import torch
import time
import numpy as np
from math import log10
from visualization import save_tensor_image
from networks.utils import classvsr, flow

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

            SR_list = []

            for i in range(t):

                if i == 0:
                    lq = torch.stack((LQ_all[:, i], LQ_all[:, i], LQ_all[:, i], LQ_all[:, i+1], LQ_all[:, i+2]), dim=1)
                elif i == 1:
                    lq = torch.stack((LQ_all[:, i-1], LQ_all[:, i-1], LQ_all[:, i], LQ_all[:, i+1], LQ_all[:, i+2]), dim=1)
                elif i == t - 2:
                    lq = torch.stack((LQ_all[:, i-2], LQ_all[:, i-1], LQ_all[:, i], LQ_all[:, i+1], LQ_all[:, i+1]), dim=1)
                elif i == t - 1:
                    lq = torch.stack((LQ_all[:, i-2], LQ_all[:, i-1], LQ_all[:, i], LQ_all[:, i], LQ_all[:, i]), dim=1)
                else:
                    lq = LQ_all[:, i - 2 : i + 3]

                lr = []
                n, t, c, h, w = lq.size()

                for i in range(t):
                    LR = lq[:, i].squeeze().permute(1, 2, 0)

                    lr_list, num_h, num_w, H, W = classvsr.crop_cpu(LR, opt.Crop, opt.Step, opt.device)

                    lr.append(lr_list)

                for i in range(len(lr)):
                    lr[i] = torch.stack(lr[i]).permute(0, 3, 1, 2)

                lr = torch.stack(lr, 1)

                A, B, C = [], [], []

                for i in range(len(lr)):
                    LQ = lr[i].unsqueeze(0)

                    dis_flow1 = flow.dis_flow(LQ[: ,1].squeeze(), LQ[: ,2].squeeze())
                    dis_flow2 = flow.dis_flow(LQ[:, 3].squeeze(), LQ[:, 2].squeeze())

                    if (dis_flow1 < opt.Threshold and dis_flow2 < opt.Threshold) :

                        A.append(i)

                    else:

                        dis_flow3 = flow.dis_flow(LQ[:, 0].squeeze(), LQ[:, 1].squeeze())
                        dis_flow4 = flow.dis_flow(LQ[:, 3].squeeze(), LQ[:, 4].squeeze())

                        if (dis_flow3 < opt.Threshold and dis_flow4 < opt.Threshold):
                            B.append(i)
                        else:
                            C.append(i)

                n, t, c, h, w = lr.size()

                if A != []:
                    lq_A = torch.stack([lr[i] for i in A])
                    output_A = model(lq_A, 'A')

                if B != []:
                    lq_B = torch.stack([lr[i] for i in B])
                    output_B = model(lq_B, 'B')

                if C != []:
                    lq_C = torch.stack([lr[i] for i in C])
                    output_C = model(lq_C, 'C')

                output = []
                for i in range(len(lr)):
                    if i in A:
                        output.append(output_A[A.index(i)])
                    elif i in B:
                        output.append(output_B[B.index(i)])
                    else:
                        output.append(output_C[C.index(i)])

                output = torch.stack(output, 0)

                sr = []

                for i in range(n):
                   sr.append(output[i].permute(1, 2, 0))

                sr = classvsr.combine(sr, num_h, num_w, H, W, opt.Crop, opt.Step, 4)

                n, t, c, h, w = LQ_all.size()

                SR_ref = sr[0:h*4, 0:w*4].permute(2, 0, 1).unsqueeze(0)

                SR_list.append(SR_ref)

            SR_ref = torch.stack(SR_list, dim=1)

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