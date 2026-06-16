import argparse
import os
from pathlib import Path
from tqdm import tqdm
from aggregation.utils.aedat4_to_img import aedat4_to_img_frame


def arg_parser():
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
    parser.add_argument('--width',
                        type=int,
                        default=346,
                        help='The width of the output image.')
    parser.add_argument('--height',
                        type=int,
                        default=260,
                        help='The height of the output image.')
    parser.add_argument('--folder_name',
                        default= 'imgs_frame',
                        type=str,
                        help='The name of the folder containing the converted images.')
    parser.add_argument('--sub_div',
                        default= 1,
                        type=int,
                        help='The number of sub-divisions of the output image.')
    return parser.parse_args()

def main(args):
    all_roots = []
    for root, dirs, files in os.walk(args.dataset_dir):  # 递归遍历数据集
        root = Path(root)
        aedat4_paths = list(root.glob('*.aedat4'))  # 找到所有aedat4文件
        if aedat4_paths:
            all_roots.append((root, aedat4_paths))
    for idx, (root, aedat4_paths) in enumerate(tqdm(all_roots), 1):
        if args.copy:  # 如果需要复制到新的目录,则创建和原数据集结构相同的输出文件夹
            rel_dir = root.relative_to(args.dataset_dir)
            out_dir = Path(args.out_dir) / rel_dir
            out_dir.mkdir(exist_ok=True, parents=True)
        else:
            out_dir = root  # 不复制,直接在原目录
        # convert the aedat4 files
        for aedat4_path in aedat4_paths:
            aedat4_to_img_frame(aedat_path=aedat4_path,
                                offset_path=args.offset_path,
                                out_dir=out_dir,
                                width=args.width,
                                height=args.height,
                                folder_name=args.folder_name,
                                sub_div=args.sub_div)

if __name__ == '__main__':
    args = arg_parser()
    main(args)