from __future__ import print_function
import torch.optim as optim
import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from importlib import import_module
import random
import re
import time
import numpy as np
from math import log10
import torch.nn.functional as F
import math
from losses.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR

# from networks.warp_image import estimate, Backward
from networks.SpyNet import flow_warp
import visualization as vl
from utils.dist_util import reduce_tensor
from visualization import save_tensor_image, show_flow

import utils.classvsr as classvsr
from thop import profile

def checkpoint(step, epoch, model, moduleNetwork, opt):
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    model_out_path = join(models_folder, "{0}/IconVSR_epoch_{1:02d}.pkl".format(step, epoch))
    torch.save(model.state_dict(), model_out_path)
    print("===>IconVSR Checkpoint saved to {}".format(model_out_path))

    model_out_path = 'models/{}/{}/network-finetuned_{:02d}.pytorch'.format(opt.name, step, epoch)
    torch.save(moduleNetwork.state_dict(), model_out_path)
    print("===>Flow Checkpoint saved to {}".format(model_out_path))

def save_training_state(step, epoch, optimizer, optimizer_pwc, schedulers, opt):
    """Save training states during training, which will be used for
    resuming.

    Args:
        epoch (int): Current epoch.
        current_iter (int): Current iteration.
    """
    root_folder = os.path.abspath('.')
    save_path = join(root_folder, 'models', opt.name, "{0}/Training_{1:02d}.state".format(step, epoch))
    if opt.current_iter != -1:
        state = {
            'epoch': epoch,
            'iter': opt.current_iter,
            'optimizer': [],
            'optimizer_pwc': [],
            'schedulers': []
        }
        state['optimizer'].append(optimizer.state_dict())
        state['optimizer_pwc'].append(optimizer_pwc.state_dict())
        for s in schedulers:
            state['schedulers'].append(s.state_dict())
        torch.save(state, save_path)

def resume_training(resume_state, optimizer, optimizer_pwc, schedulers):
    """Reload the optimizers and schedulers for resumed training.

    Args:
        resume_state (dict): Resume state.
    """
    resume_optimizer = resume_state['optimizer']
    resume_optimizer_pwc = resume_state['optimizer_pwc']
    resume_schedulers = resume_state['schedulers']

    assert len(resume_schedulers) == len(
        schedulers), 'Wrong lengths of schedulers'
    optimizer.load_state_dict(resume_optimizer[0])
    optimizer_pwc.load_state_dict(resume_optimizer_pwc[0])
    for i, s in enumerate(resume_schedulers):
        schedulers[i].load_state_dict(s)

def gradient_loss(pred, GT, kernel_size=15):
    pad_size = (kernel_size - 1) // 2
    pred = pred.sum(1)
    GT = GT.sum(1)
    BN = pred.size()[0]
    M = pred.size()[1]
    N = pred.size()[2]
    pred_pad = torch.nn.ZeroPad2d(pad_size)(pred)
    GT_pad = torch.nn.ZeroPad2d(pad_size)(GT)

    gradient_loss = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == pad_size and j == pad_size:
                continue
            data = pred
            neighbour = pred_pad[:, i:M + i, j:N + j]
            pred_gradient = data - neighbour

            data = GT
            neighbour = GT_pad[:, i:M + i, j:N + j]
            GT_gradient = data - neighbour

            gradient_loss = gradient_loss + (pred_gradient - GT_gradient).abs().sum()

    return gradient_loss / (BN * 3 * M * N * (kernel_size ** 2 - 1))


def train(trainloader, model, moduleNetwork, criterion, optimizer, optimizer_pwc, schedulers, epoch, opt):
    epoch_loss = 0
    part_time_gpu = time.perf_counter()

    for iteration, batch in enumerate(trainloader, 1):
        if opt.verbose:
            print('rank is {} and pre_iteration is {}\n'.format(opt.rank_index, opt.current_iter))
        opt.current_iter += 1
        if opt.verbose:
            torch.distributed.barrier()
            print('rank is {} and after_iteration is {}\n'.format(opt.rank_index, opt.current_iter))
        if opt.current_iter >= opt.total_iters:
            break
        if opt.current_iter >= 5000:
            opt.tof = True
        if opt.current_iter > 1:
            for scheduler in schedulers:
                scheduler.step()
        # adjust_learning_rate(opt.current_iter, optimizer, optimizer_pwc, opt)

        # input, targetdeblur, targetsr
        LQ_all = batch[0].to(opt.device)
        # Flow_ALL = batch[1].to(opt.device)
        GT_all = batch[2].to(opt.device)

        n, t, c, h, w = LQ_all.size()

        if not opt.tof:
            with torch.no_grad():
                moduleNetwork.eval()
                lrs_1 = LQ_all[:, :-1, :, :, :].reshape(-1, c, h, w)
                lrs_2 = LQ_all[:, 1:, :, :, :].reshape(-1, c, h, w)

                flows_backward = moduleNetwork(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
                flows_forward = moduleNetwork(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        else:
            moduleNetwork.train()
            lrs_1 = LQ_all[:, :-1, :, :, :].reshape(-1, c, h, w)
            lrs_2 = LQ_all[:, 1:, :, :, :].reshape(-1, c, h, w)

            flows_backward = moduleNetwork(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
            flows_forward = moduleNetwork(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        SR_Results = model(LQ_all, flows_backward, flows_forward)

        if len(SR_Results.shape) == 5:
            b, n, c, h, w = SR_Results.shape
            SR_Results = SR_Results.view(b * n, c, h, w).contiguous()
            GT_all = GT_all.view(b * n, c, h, w).contiguous()

        if opt.hem:
            if opt.vgg:
                perceptual_loss = opt.loss_network(SR_Results, GT_all)
            # loss_time = time.perf_counter()
            mask = hem_loss(SR_Results, GT_all) * 0.5 + 1

            loss = criterion(SR_Results * mask, GT_all * mask)
            mse = loss  # + opt.lambda_db * loss1
            mse = mse + 0.1 * perceptual_loss
            # loss = criterion(Dehazed, GT)
            # torch.cuda.synchronize()
            # loss_time_gpu = time.perf_counter() - loss_time
            # print("===> Part Time: {:.6f}".format(loss_time_gpu))
        else:
            mse = criterion(SR_Results, GT_all)
            loss = mse  # + opt.lambda_db * loss1
            if opt.vgg:
                SR_fea = opt.vgg_model(SR_Results)
                GT_fea = opt.vgg_model(GT_all)
                perceptual_loss = opt.cri_fea(SR_fea, GT_fea)
                loss = mse + opt.fea_weight * perceptual_loss

        # reduce loss and measurement between GPUs
        if opt.verbose:
            print('rank is {} and part_loss is {}\n'.format(opt.rank_index, mse.item()))
            torch.distributed.barrier()
        reduced_loss = reduce_tensor(opt.world_size, opt.rank_index, mse.detach())
        if opt.verbose and opt.rank_index == 0:
            print('rank is {} and reduced_loss is {}\n'.format(opt.rank_index, reduced_loss.item()))
        epoch_loss += reduced_loss.item()

        # backward and optimize
        optimizer.zero_grad()
        optimizer_pwc.zero_grad()
        loss.backward()
        optimizer.step()
        if opt.tof:
            optimizer_pwc.step()

        if opt.current_iter % 500 == 0 and opt.rank_index == 0:
            print("===> Epoch[{}] Iteration[{}]: MSE Loss{:.4f};".format(epoch, opt.current_iter, reduced_loss.item()))
        if opt.verbose:
            break
    # print("===>Avg MSE loss is :{:4f}".format(epoch_loss / len(trainloader)))
    part_eval_time_gpu = time.perf_counter() - part_time_gpu
    # print("===> Part Time: {:.6f}".format(part_eval_time_gpu))
    return epoch_loss / len(trainloader), part_eval_time_gpu


def test(testloader, model, criterion, opt):
    avg_psnr = 0
    num = 0
    psnrs_folder = []
    H_forward_init = None

    with torch.no_grad():
        for iteration, batch in enumerate(testloader, 1):
            # input, targetdeblur, targetsr
            if not opt.isReal:
                LQ_all = batch[0].to(opt.device)
                # Flow_ALL = batch[1].to(opt.device)
                GT_all = batch[2].to(opt.device)
                names = batch[3]
                dir_name = batch[4]
            else:
                LQ_all = batch[0].to(opt.device)
                names = batch[1]
                dir_name = batch[2]

            # HR = F.interpolate(LR, scale_factor=4, mode='bicubic', align_corners=False)
            H_ref_back_all = []
            E_ref_all = []
            key_frames = []

            # HR = F.interpolate(LR, scale_factor=4, mode='bicubic', align_corners=False)

            # Backward
            start_time = time.perf_counter()  # -------------------------begin to deal with the video

            n, t, c, h, w = LQ_all.size()

            # random_frame = 50
            #
            # number = []
            # for i in range(random_frame):
            #     n, t, c, h, w = LQ_all.size()
            #     a = random.randint(0, t - 2)
            #     LQ_all = torch.cat((LQ_all[:, 0:a + 1], LQ_all[:, a: a + 1], LQ_all[:, a + 1:t]), dim=1)
            #     number.append(a)
            #
            # number = number[::-1]

            SR_ref = model(LQ_all)

            # for i in range(random_frame):
            #     n, t, c, h, w = SR_ref.size()
            #     a = number[i]
            #     SR_ref = torch.cat((SR_ref[:,0:a+1], SR_ref[:,a+2:t]), dim=1)
            #
            # n, t, c, h, w = SR_ref.size()

            # video_name = dir_name[0].split('/')[-1]  # name for test video
            # root = os.path.join(
            #     '/data/hyh/VSR/class_test/flow_forward_results2',
            #     video_name)
            # if not os.path.exists(root):
            #     os.makedirs(root)
            #
            # for i in range(t - 1):
            #     frame_index = iteration - 1 + i
            #     frame_name = names[frame_index][0]
            #     path = os.path.join(root, frame_name)
            #     flows = flows_forward[:, i]
            #     show_flow(flows, path)
            #
            # video_name = dir_name[0].split('/')[-1]  # name for test video
            # root = os.path.join(
            #     '/data/hyh/VSR/class_test/flow_backward_results2',
            #     video_name)
            # if not os.path.exists(root):
            #     os.makedirs(root)
            #
            # for i in range(t - 1):
            #     frame_index = iteration - 1 + i
            #     frame_name = names[frame_index][0]
            #     path = os.path.join(root, frame_name)
            #     flows = flows_backward[:, i]
            #     show_flow(flows, path)

            # with open("flops.txt", "a") as f:
            #
            #     from networks.SpyNet import SpyNet as Network
            #     net1 = Network().to(opt.device)
            #     Net = import_module('networks.' + opt.model + '_arch')
            #     net2 = Net.make_model(opt).to(opt.device)
            #
            #     flops1, params1 = profile(net1, inputs=(lrs_1, lrs_2))
            #     flops2, params2 = profile(net2, inputs=(LQ_all, flows_backward, flows_forward))
            #
            #     flops = flops1 * 2 + flops2
            #     params = params1 + params2
            #     print(flops / 1e9, params / 1e6)  # flops单位G，para单位M
            #     f.write("flops : %fG ; params : %fM ; " % (flops / 1e9, params / 1e6))

            # for 2X
            # SR_ref = F.interpolate(SR_ref, scale_factor=0.5, mode='bicubic')
            # LQ_ref_2x = F.interpolate(LQ_ref, scale_factor=2, mode='bicubic')
            # loss1 = criterion(lr_deblur, LR_Deblur) * (max(15-epoch, 0) / 15)
            # LQ_ref_2x_clipped = torch.clamp(LQ_ref_2x, 0, 1)
            ################### save image ###################
            if opt.isTest:
                SR_ref_clipped = torch.clamp(SR_ref, 0, 1)

                video_name = dir_name[0].split('/')[-1]  # name for test video

                root = os.path.join(
                    '/data/hyh/VSR/class_test/BasicVSR_results_class/fra5',
                    video_name)
                # root = os.path.join('/data/hyh/VSR/ClassVSR/Class4K_scale/sub_test_C/BasicVSR_results/', video_name)
                if not os.path.exists(root):
                    os.makedirs(root)

                # i = t // 2

                for i in range(t):
                    frame_index = iteration - 1 + i
                    frame_name = names[frame_index][0]
                    # p = names.index(('00000006.png',))
                    # frame_name = '00000006.png'
                    path = os.path.join(root, frame_name)

                    if (i + 1) % 3 == 0:
                        save_tensor_image(SR_ref_clipped[:, i], path)

            for i in range(t):
                ################### calculate psnr ###################
                if not opt.isReal:

                    # SR = bgr2ycbcr(SR_ref[:,i].squeeze().permute(2, 1, 0).cpu().numpy())
                    # GT = bgr2ycbcr(GT_all[:,i].squeeze().permute(2, 1, 0).cpu().numpy())
                    #
                    # img1 = SR.astype(np.float64)
                    # img2 = GT.astype(np.float64)
                    # mse = np.mean((img1 - img2) ** 2)
                    # if mse == 0:
                    #     return float('inf')
                    #
                    # psnr = 20 * math.log10(1 / math.sqrt(mse))

                    mse = criterion(SR_ref[:,i], GT_all[:,i])
                    if mse == float("inf"):
                        mse = 1

                    psnr = 10 * log10(1 / mse)
                    # print('{}'.format(psnr))
                    # print("Processing {},{}".format(iteration, psnr))
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
    # print(opt)
    psnr = test(testloader, model, criterion, opt)
    return psnr

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
