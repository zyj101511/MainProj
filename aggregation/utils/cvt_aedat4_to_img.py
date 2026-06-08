import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dv import AedatFile


def _make_color_img(events, width, height):
    canvas = 255 * np.ones((height, width, 3), dtype=np.uint8)  # 0 ~ 255
    if events.size:
        assert events['x'].max() < width, f"out of bound event: x = {events['x'].max()}, should less than: width = {width}"
        assert events['y'].max() < height, f"out of bound event: y = {events['y'].max()}, should less than: height = {height}"

        on_idxs = np.where(events['polarity'] == 1)[0]
        canvas[events['y'][on_idxs], events['x'][on_idxs], :] = [0, 0, 255]

        off_idxs = np.where(events['polarity'] == 0)[0]
        canvas[events['y'][off_idxs], events['x'][off_idxs], :] = [255, 0, 0]
    return canvas


def cvt_aedat4_to_img(aedat_path, event_num, out_dir, width, height):

    with AedatFile(str(aedat_path)) as f:
        events = np.hstack([event for event in f['events'].numpy()])
        # print(events[0].dtype)  # (timestamp, x, y, polarity, _p1, _p2)
        event_num = event_num
        frame_idx = 0

        output_dir = Path(out_dir)
        output_dir.mkdir(exist_ok=True)

        img_dir = output_dir / 'aedat4_imgs'
        img_dir.mkdir(exist_ok=True)

        # saving the frame_idxs and timestamps for frame images.
        ts_list = []

        for event_idx in tqdm(range(0, len(events), event_num), leave=False):
            rec_events = events[event_idx:event_idx+event_num]
            event_img = _make_color_img(rec_events, width=width, height=height)
            frame_idx += 1

            img_path = img_dir / f'frame_{frame_idx:06d}.png'
            cv2.imwrite(str(img_path), event_img)
            # cv2.imshow('event_img', event_img)
            # cv2.waitKey(500)

            ts_list.append([frame_idx, events['timestamp'][event_idx]])

        ts_df = pd.DataFrame(ts_list, columns=['frame_idx', 'timestamp'])
        ts_df.to_csv(output_dir / 'aedat4_img_timestamp.csv', index=False)

if __name__ == '__main__':
    aedat_path = '/media/yanjiezhang/ian/dataset/FE108_raw/train/dove/events.aedat4'
    event_num = 10000
    out_dir = '/home/yanjiezhang/Downloads/cc/test'
    width, height = 346, 260
    cvt_aedat4_to_img(aedat_path, event_num, out_dir, width, height)