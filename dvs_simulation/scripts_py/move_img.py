from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

def processing(img_dir, out_dir, file_type):
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)

    all_roots = []

    for root in img_dir.rglob("*"):  # 返回所有item, 有文件夹有文件
        if root.is_dir():  # 只查看文件夹
            img_paths = list(root.glob(f"*.{file_type}"))
            if img_paths:
                all_roots.append(root)

    for root in tqdm(all_roots, desc="Moving jpg files"):
        rel_dir = root.relative_to(img_dir)
        tar_dir = out_dir / rel_dir
        tar_dir.mkdir(parents=True, exist_ok=True)

        for item in root.iterdir():
            src_path = item
            dst_path = tar_dir / item.name
            shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_dir', required=True, help='path to the original dataset')
    parser.add_argument('--out_dir', required=True, help='path to the new dataset')
    parser.add_argument('--file_type', default='jpg', type=str, help='path to the new dataset')

    args = parser.parse_args()
    processing(args.img_dir, args.out_dir, args.file_type)

if __name__ == "__main__":
    main()


