import argparse
import os
from pathlib import Path
from tqdm import tqdm
from aggregation.utils.aedat4_to_voxel import aedat4_to_voxel

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset_dir',
                    type=str,
                    required=True,
                    help='Path to the dataset directory containing AEDAT files.')
parser.add_argument('-f', '--offset_path',
                    type=str,
                    default=None,
                    help='Path to the offset file.')
parser.add_argument('-c', '--copy',
                    action='store_true')
parser.add_argument('-o', '--out_dir',
                    default=None,
                    type=str,
                    help='Path to the output directory if copy=True.')
parser.add_argument('-n', '--num_bins',
                    type=int,
                    default=5,
                    help='The number of bins in each voxel.')
parser.add_argument('--width',
                    type=int,
                    default=346,
                    help='The width of the output image.')
parser.add_argument('--height',
                    type=int,
                    default=260,
                    help='The height of the output image.')


def main(arg):
    all_roots = []
    for root, dirs, files in os.walk(arg.dataset_dir):  # 递归遍历数据集
        root = Path(root)
        aedat4_paths = list(root.glob('*.aedat4'))  # 找到所有aedat4文件
        if aedat4_paths:
            all_roots.append((root, aedat4_paths))
    for idx, (root, aedat4_paths) in enumerate(tqdm(all_roots), 1):
        if arg.copy:  # 如果需要复制到新的目录,则创建和原数据集结构相同的输出文件夹
            rel_dir = root.relative_to(arg.dataset_dir)
            out_dir = Path(arg.out_dir) / rel_dir
            out_dir.mkdir(exist_ok=True, parents=True)
        else:
            out_dir = root  # 不复制,直接在原目录
        # convert the aedat4 files
        for aedat4_path in aedat4_paths:
            aedat4_to_voxel(aedat4_path, arg.offset_path, out_dir, arg.width, arg.height, num_bins=arg.num_bins)


if __name__ == '__main__':
    main(parser.parse_args())