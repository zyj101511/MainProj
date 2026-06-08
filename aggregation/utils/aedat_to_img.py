import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dv import AedatFile


parser = argparse.ArgumentParser()

parser.add_argument('-p', '--abs_path_to_aedat',
                    type=str,
                    required=True,
                    help='Absolute Path to an AEDAT file.')
parser.add_argument('-n', '--event_num',
                    type=int,
                    default=15000,
                    help='The number of events for each frame image.')
parser.add_argument('-o', '--output_dir',
                    type=str,
                    default='./output',
                    help='The directory to store the output images and txt file.')
parser.add_argument('--width',
                    type=int,
                    default=346,
                    help='The width of the output image.')
parser.add_argument('--height',
                    type=int,
                    default=260,
                    help='The height of the output image.')


def make_color_frame(events, width, height):
    canvas = 255 * np.ones((height, width, 3), dtype=np.uint8)  # 0 ~ 255
    if events.size:
        assert events['x'].max() < width, f"out of bound event: x = {events['x'].max()}, should less than: width = {width}"
        assert events['y'].max() < height, f"out of bound event: y = {events['y'].max()}, should less than: height = {height}"

        on_idxs = np.where(events['polarity'] == 1)[0]
        canvas[events['y'][on_idxs], events['x'][on_idxs], :] = [0, 0, 255]

        off_idxs = np.where(events['polarity'] == 0)[0]
        canvas[events['y'][off_idxs], events['x'][off_idxs], :] = [255, 0, 0]
    return canvas


def main(args):
    input_path = args.abs_path_to_aedat

    with AedatFile(input_path) as f:
        events = np.hstack([event for event in f['events'].numpy()])
        # print(events[0].dtype)  # (timestamp, x, y, polarity, _p1, _p2)
        event_num = args.event_num
        frame_idx = 0

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        img_dir = output_dir / 'imgs'
        img_dir.mkdir(exist_ok=True)

        # saving the frame_idxs and timestamps for frame images.
        ts_list = []

        for event_idx in tqdm(range(0, len(events), event_num)):
            rec_events = events[event_idx:event_idx+event_num]
            event_img = make_color_frame(rec_events, width=args.width, height=args.height)
            frame_idx += 1

            img_path = img_dir / f'frame_{frame_idx:06d}.png'
            cv2.imwrite(str(img_path), event_img)
            # cv2.imshow('event_img', event_img)
            # cv2.waitKey(500)

            ts_list.append([frame_idx, events['timestamp'][event_idx]])

        ts_df = pd.DataFrame(ts_list, columns=['frame_idx', 'timestamp'])
        ts_df.to_csv(output_dir / 'timestamp.csv', index=False)

if __name__ == '__main__':
    main(parser.parse_args())