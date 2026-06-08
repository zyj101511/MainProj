import os
import argparse
from glob import glob
import shutil
from tqdm import tqdm


def processing(img_dir, mp4_dir):
    all_roots = []
    for root, dirs, files in os.walk(img_dir):  # 递归遍历数据集
        txt_paths = glob(os.path.join(root, "*.txt"))  # 找到所有txt文件
        if txt_paths:
            all_roots.append((root, txt_paths))
    for idx, (root, txt_paths) in enumerate(tqdm(all_roots, desc="Moving txt files"), 1):
        rel_dir = os.path.relpath(root, img_dir)
        out_dir = os.path.join(mp4_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)  # 创建和原数据集结构相同的输出文件夹
        # 复制所有txt
        for txt_path in txt_paths:
            out_txt_path = os.path.join(out_dir, os.path.basename(txt_path))
            shutil.copy(txt_path, out_txt_path)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_dir', required=True, help='path to the original dataset')
    parser.add_argument('--mp4_dir', required=True, help='path to the new dataset')

    args = parser.parse_args()
    processing(args.img_dir, args.mp4_dir)

if __name__ == "__main__":
    main()


