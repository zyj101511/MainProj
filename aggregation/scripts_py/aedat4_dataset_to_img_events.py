import argparse
import os
from pathlib import Path
from tqdm import tqdm
from aggregation.utils.aedat4_to_img import aedat4_to_img_events

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset_dir',
                    type=str,
                    required=True,
                    help='Path to the dataset directory containing AEDAT files.')
parser.add_argument('-n', '--event_num',
                    type=int,
                    default=15000,
                    help='The number of events for each frame image.')
parser.add_argument('-c', '--copy',
                    action='store_true')
parser.add_argument('-o', '--out_dir',
                    default=None,
                    type=str,
                    help='Path to the output directory if copy=True.')
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
            aedat4_to_img_events(aedat4_path, arg.event_num, out_dir, arg.width, arg.height)


if __name__ == '__main__':
    main(parser.parse_args())