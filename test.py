from __future__ import print_function
import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from importlib import import_module
import numpy as np
from datasets.dataset_REDS import DataValSetRNN

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")

# Dataset settings
parser.add_argument('--dataset', default="C:\Datasets\VideoHazy", type=str, help='Path of the training dataset')
parser.add_argument('--dataset_test', default="/data/hyh/VSR/DTVIT", type=str, help='Path of the test dataset')
parser.add_argument('--save_path', default="/data/hyh/DTVIT_result", type=str, help='Path of the test result')
parser.add_argument('--port', default="9802", type=str, help='')
parser.add_argument('--flow_dir', default="hazy", type=str, help='Path of the training dataset(.h5)')
parser.add_argument("--scale", default=4, type=int, help="Scale factor, Default: 4")
parser.add_argument("--cropSize", type=int, default=64, help="LR patch size")
parser.add_argument("--frames", type=int, default=5, help="the amount of input frames")
parser.add_argument("--repeat", type=int, default=1, help="the amount of the dataset repeat per epoch")
parser.add_argument("--frame_batch", type=int, default=-1, help="the amount of input frames")
parser.add_argument("--frame_batch_test", type=int, default=-1, help="the amount of input frames")
parser.add_argument("--iteration_test", type=int, default=-1, help="the amount of input frames")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--data_type", type=str, default='REDS', help="Start epoch from 1")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
parser.add_argument("--isReal", type=bool, default=False, help="Real Video or not")
parser.add_argument("--isSave", type=bool, default=False, help="Save or not")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")

# Network settings
parser.add_argument('--model', default='Boosted_BasicVSR', type=str, help='Import which network')
parser.add_argument('--suffix', default='baseline_LR_flow', type=str, help='Filename of the training models')
parser.add_argument("--resume",
                    default="/home/huangyuhao/projects/VideoSR/github/Boosted_BasicVSR/models/basicvsr_dtvit-traint2.pth",
                    type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--n_feats", type=int, default=64, help="the amount of input frames")
parser.add_argument("--residual", type=str2bool, nargs='?', const=True, help="Activated hr_flow")

# Flowchart settings
parser.add_argument("--pre", default=0, type=int,
                    help="Ways of Pre-Dehazing Module, 0: No Pre-Dehazing / 1: Pre-Dehazing / 2: Pre-Dehazing and Finetune")
parser.add_argument("--warped", default=1, type=int,
                    help="Ways of Alignment, 0: No Alignment / 1: Input Alignment / 2: Feature Alignment / 3: Feature-based flow")
parser.add_argument("--flow_method", type=str, default='SPY',
                    help='Which method to calculate the optical flow. [SPY | PWC]')
parser.add_argument("--tof", type=bool, default=False, help="Activated PWC-Net finwtuning")
parser.add_argument("--hr_flow", type=bool, default=False, help="Activated hr_flow")

# Learning settings
parser.add_argument("--cos_adj", type=bool, default=False, help="Using cosine annealing strategy?")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate, default=1e-5")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--lambda_GL", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")

# Loss settings
parser.add_argument("--cobi", type=bool, default=False, help="Use CoBi Loss")
parser.add_argument("--vgg", type=bool, default=False, help="Activated vgg loss")
parser.add_argument("--hem", type=bool, default=False, help="Activated hard negative mining")
parser.add_argument("--reduction", type=str, default='mean',
                    help='Which method to calculate the optical flow. [sum | mean]')

# ddp settings
parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0, 1, 2. use -1 for CPU')
parser.add_argument("--para_mode", type=str, default='DDP', help='Which parallele method. [DP | DDP]')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, help="Display debug information or not")

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='A',
                    help='parameters config of RDN. (Use in RDN)')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

# ClassVSR
parser.add_argument("--Crop", type=int, default=64, help="Size of crop")
parser.add_argument("--Step", type=int, default=56, help="step of crop")
parser.add_argument("--Threshold", type=float, default=0.2, help="step of crop")

def Process(opt):

    # create model
    print("===> Loading model {} and criterion".format(opt.model))
    SRmodel = import_module(opt.model)
    Net = import_module('networks.' + opt.model + '_arch')
    print('Model is {}'.format(opt.model))

    if opt.gpu_ids == '-1':
        opt.device = torch.device('cpu')
    else:
        opt.device = torch.device('cuda:{}'.format(opt.gpu_ids))

    print("Loading VSR model from checkpoint {}".format(opt.resume))
    model = Net.make_model(opt)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(opt.resume)
    for key in model_dict.keys():
        model_dict[key] = pretrained_dict[key]
    model.load_state_dict(model_dict)

    model = model.to(opt.device)

    print(opt)

    #######################################  Test  ######################################
    if not opt.isReal:
        test_dir = opt.dataset_test
        test_path = test_dir
        input_dir_test = join(test_path, 'LR')
        gt_dir_test = join(test_path, 'HR')
        test_sets = [x for x in sorted(os.listdir(gt_dir_test))]
        print('Evaluation dataset: [GT] {}, [LR] {}'.format(gt_dir_test, input_dir_test))
    else:
        input_dir_test = opt.dataset_test
        gt_dir_test = opt.dataset_test
        test_sets = [x for x in sorted(os.listdir(input_dir_test))]
        print('Evaluation real dataset: [LR] {}'.format(input_dir_test))

    opt.psnrs_all = []
    for j in range(len(test_sets)):
        print("Folder {}: Input folder is {}, GT folder is {}".format(j + 1,
                                                                      join(input_dir_test, test_sets[j]),
                                                                      join(gt_dir_test, test_sets[j])))
        test_set = DataValSetRNN(join(input_dir_test, test_sets[j]),
                                 join(gt_dir_test, test_sets[j]), opt)

        testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, pin_memory=False,
                                num_workers=0)

        SRmodel.model_test(testloader, model, opt)

    print("Testing Avg. PSNR is :{:4f}".format(np.mean(opt.psnrs_all)))

if __name__ == '__main__':
    opt = parser.parse_args()
    Process(opt)
    print("Test stage is done")