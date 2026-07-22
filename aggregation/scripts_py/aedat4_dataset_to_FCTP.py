import os
import argparse
from tqdm import tqdm
from pathlib import Path
from aggregation.utils.aedat4_to_FCTP import aedat4_to_FCTP


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--copy',
                        action='store_true',
                        help='True if dataset is needed to be move to another folder')
    parser.add_argument('-d', '--dataset_dir',
                        type=str,
                        required=True,
                        help='Path to the dataset directory containing AEDAT files.')
    parser.add_argument('-f', '--offset_path',
                        type=str,
                        default=None,
                        help='Path to the offset file.')
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
                        default='imgs_FCTP',
                        type=str,
                        help='The name of the folder containing the converted images.')
    parser.add_argument('--ch12_strength', type=float, default=40,
                        help='Value of single stacking for the first two channels (positive, negative)')

    parser.add_argument('--ch3_strength', type=float, default=30,
                        help='Value of single stacking for the third channel (hidden)')

    parser.add_argument('--agg_decay_rate', type=float, default=0.1,
                        help='Decay of values between two frames for the aggregation of bins')

    parser.add_argument('--ch12_decay_rate', type=float, default=0.,
                        help='Decay of values between two frames for the first two channels (positive, negative)')

    parser.add_argument('--ch3_decay_rate', type=float, default=0.7,
                        help='Decay of values between two frames for the third channel (hidden)')
    parser.add_argument('--num_bins', type=int, default=4,
                        help='Number of bins inside one original frame interval')

    parser.add_argument('--output_mode', type=str, default='per_bin',
                        choices=['aggregate', 'per_bin', 'both'],
                        help='Output mode for the converted images. Options: "aggregate", "per_bin", "both".')

    return parser.parse_args()


def main(args):
    all_roots = []
    for root, dirs, files in os.walk(args.dataset_dir):
        root = Path(root)
        aedat4_paths = list(root.glob('*.aedat4'))
        if aedat4_paths:
            all_roots.append((root, aedat4_paths))

    for idx, (root, aedat4_paths) in enumerate(tqdm(all_roots), 1):
        if args.copy:
            rel_dir = root.relative_to(Path(args.dataset_dir))
            out_dir = Path(args.out_dir) / rel_dir
            out_dir.mkdir(exist_ok=True, parents=True)
        else:
            out_dir = root

        for aedat4_path in aedat4_paths:
            aedat4_to_FCTP(
                aedat_path=aedat4_path,
                offset_path=args.offset_path,
                out_dir=out_dir,
                folder_name=args.folder_name,
                width=args.width,
                height=args.height,
                ch12_strength=args.ch12_strength,
                ch3_strength=args.ch3_strength,
                agg_decay_rate = args.agg_decay_rate,
                ch12_decay_rate = args.ch12_decay_rate,
                ch3_decay_rate=args.ch3_decay_rate,
                num_bins=args.num_bins,
                output_mode=args.output_mode
            )


if __name__ == '__main__':
    args = arg_parser()
    main(args)
