# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from os.path import join
import re

def reduce_tensor(num_gpus, rank, ts):
    """
    reduce tensor from multiple gpus
    """
    # todo loss of ddp mode
    if isinstance(ts, dict):
        raise NotImplementedError
    else:
        try:
            with torch.no_grad():
                dist.reduce(ts, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    ts /= num_gpus
        except:
            msg = '{}'.format(type(ts))
            raise NotImplementedError(msg)
    return ts

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def mkdir_steptraing(opt):
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    step1_folder, step2_folder, step3_folder, step4_folder = join(models_folder, '1'), join(models_folder, '2'), join(
        models_folder, '3'), join(models_folder, '4')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder) and os.path.exists(
        step3_folder) and os.path.exists(step4_folder)
    if not isexists:
        os.makedirs(step1_folder)
        os.makedirs(step2_folder)
        os.makedirs(step3_folder)
        os.makedirs(step4_folder)
        print("===> Step training models store in models/1 & /2 & /3.")


def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])


def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[-3:-2])
    start_epoch = "".join(re.findall(r"\d", resume)[-3:])
    return int(trainingstep), int(start_epoch)



def adjust_learning_rate(epoch, optimizer, optimizer_pwc, opt):
    def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
        lr = 0.01 * init_lr +  0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
        return lr
    if opt.cos_adj:
        lr = lr_schedule_cosdecay(epoch-1, opt.nEpochs)
        lr_flow = lr/8
    else:
        lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
        lr_flow = lr/8
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_pwc.param_groups:
        param_group['lr'] = lr_flow
