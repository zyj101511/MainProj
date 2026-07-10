import os
import argparse
import importlib
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
import numpy as np
from lib.settings.settings import Settings
from lib.train.train_script import run
torch.backends.cudnn.benchmark = False

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1,
                 save_dir=None, base_seed=None):

    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")

    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))

    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    settings = Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)

    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    settings.cfg_file = os.path.join(prj_dir, f'experiments/{config_name}.yaml')
    run(settings)

def main():
    parser = argparse.ArgumentParser(description='Run training script.')
    parser.add_argument('--script_name', type=str, required=True, help='Name of the training script.')
    parser.add_argument('--config_name', type=str, required=True, help='Name of the configuration file.')
    parser.add_argument('--cudnn_benchmark', action='store_true', help='Use cudnn benchmark.')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save logs and checkpoints.')
    parser.add_argument('--base_seed', type=int, default=None, help='Base seed for random number generators.')

    args = parser.parse_args()

    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)

    run_training(args.script_name, args.config_name, args.cudnn_benchmark,
                 args.local_rank, args.save_dir, args.base_seed)


if __name__ == '__main__':
    main()
