import os
import argparse
import random


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script_name', type=str, help='training script name')
    parser.add_argument('--config_name', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="single",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple

    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = f"python3 -m lib.train.run_training --script_name {args.script_name} " \
                    f"--config_name {args.config_name} --save_dir {args.save_dir} "
    elif args.mode == "multiple":
        train_cmd = f"python3 -m torch.distributed.launch --nproc_per_node {args.nproc_per_node} --master_port " \
                    f"{random.randint(10000, 50000)} lib/train/run_training.py " \
                    f"--script_name {args.script_name} --config_name {args.config_name} --save_dir {args.save_dir} "
    elif args.mode == "multi_node":
        train_cmd = f"python3 -m torch.distributed.launch --nproc_per_node {args.nproc_per_node} --master_addr {args.ip} --master_port {args.port} --nnodes {args.world_size} --node_rank {args.rank} lib/train/run_training.py " \
                    f"--script_name {args.script_name} --config_name {args.config_name} --save_dir {args.save_dir} "
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
