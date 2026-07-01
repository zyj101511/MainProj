import argparse
import json
import lmdb
from tqdm import tqdm
from pathlib import Path


def iter_files(split_dir, sub_dir_name, gt_name):
    split_dir = Path(split_dir)
    for video_dir in sorted(split_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        sub_dir = video_dir / sub_dir_name
        if not sub_dir.is_dir():
            continue

        gt_path = video_dir / gt_name
        if not gt_path.is_file():
            raise RuntimeError(f"gt_path {gt_path} does not exist")

        video_name = video_dir.name
        frame_names = sorted(p.name for p in sub_dir.iterdir() if p.is_file())
        yield video_name, sub_dir, frame_names, gt_path

def file_bytes(path):
    with open(path, 'rb') as f:
        return f.read()

def build_lmdb(src_root, dst_lmdb, sub_dir_name, gt_name, map_size=4 * 1024**3):
    src_root = Path(src_root)
    dst_lmdb = Path(dst_lmdb)
    env = lmdb.open(str(dst_lmdb), map_size=map_size)

    with env.begin(write=True) as txn:
        for split in ['train', 'test']:
            split_dir = src_root / split
            meta = {'videos': []}

            for video_name, sub_dir, frame_names, gt_path in tqdm(iter_files(split_dir, sub_dir_name, gt_name)):
                meta['videos'].append({
                    'video_name': video_name,
                    'num_frames': len(frame_names),
                    'frame_names': frame_names,
                })

                for frame_name in frame_names:
                    frame_path = sub_dir / frame_name
                    key = f'{split}/{video_name}/{frame_name}'
                    txn.put(key.encode(), file_bytes(frame_path))

                txn.put(f'{split}/{video_name}/gt.txt'.encode(), file_bytes(gt_path))

            txn.put(f'{split}/meta.json'.encode(), json.dumps(meta).encode())

    env.sync()  # 数据强制同步到磁盘
    env.close()

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', required=True, help='path to the dataset')
    parser.add_argument('--dst_lmdb', required=True, help='output path of the lmdb file')
    parser.add_argument('--sub_dir_name', required=True, type=str, help='the sub dir name in the video folders')
    parser.add_argument('--gt_name', type=str, default='groundtruth_rect.txt', help='ground truth files name')
    parser.add_argument('--map_size', default=64 * 1024 ** 3, type=int, help='maximum size of the lmdb file in bytes')
    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    build_lmdb(args.src_root, args.dst_lmdb, args.sub_dir_name, args.gt_name, map_size=args.map_size)
