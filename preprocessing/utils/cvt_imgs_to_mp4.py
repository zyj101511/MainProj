import cv2
import os
import argparse
from glob import glob
from tqdm import tqdm


def cvt_jpg_to_mp4(img_paths, out_path, fps, folder_info):
    first_frame = cv2.imread(img_paths[0])
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for path in tqdm(img_paths, desc=f"Folder: {folder_info} | Converting", leave=False):
        img = cv2.imread(path)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()

def processing(img_dir, mp4_dir, fps, img_format):
    all_roots = []
    for root, dirs, files in os.walk(img_dir):  # 递归遍历数据集
        img_paths = sorted(glob(os.path.join(root, f"*.{img_format}")))  # 按文件名排序.jpg文件
        if img_paths:
            all_roots.append((root, img_paths))
    total_roots = len(all_roots)
    for idx, (root, imgs_path) in enumerate(all_roots, 1):
        rel_dir = os.path.relpath(root, img_dir)
        out_dir = os.path.dirname(os.path.join(mp4_dir, rel_dir))
        os.makedirs(out_dir, exist_ok=True)  # 创建和原数据集结构相同的输出文件夹

        out_path = os.path.join(out_dir, "raw.mp4")
        folder_info = f"{idx}/{total_roots}"
        cvt_jpg_to_mp4(img_paths, out_path, fps, folder_info)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_dir', required=True, help='path to the original dataset')
    parser.add_argument('--mp4_dir', required=True, help='output path of the mp4 files')
    parser.add_argument('--fps', required=True, type=int, help='the fps of the output mp4 filde')
    parser.add_argument('--format', required=True, type=str, default='jpg', help='image format')
    args = parser.parse_args()
    processing(args.img_dir, args.mp4_dir, args.fps, args.format)

if __name__ == "__main__":
    main()


