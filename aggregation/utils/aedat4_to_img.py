import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dv import AedatFile

'''
def _make_color_img(events, width, height):
    """
    create a color image from events, where on events are in red and off events are in blue, and the background is white.
    """
    canvas = 255 * np.ones((height, width, 3), dtype=np.uint8)  # 0 ~ 255
    # last event
    if events.size:
        assert events['x'].max() < width, f"out of bound event: x = {events['x'].max()}, should less than: width = {width}"
        assert events['y'].max() < height, f"out of bound event: y = {events['y'].max()}, should less than: height = {height}"

        for e in events:
            if e['polarity'] == 1:
                canvas[e['y'], e['x']] = [0, 0, 255]
            else:
                canvas[e['y'], e['x']] = [255, 0, 0]
    # off overwrites on
    if events.size:
        assert events['x'].max() < width, f"out of bound event: x = {events['x'].max()}, should less than: width = {width}"
        assert events['y'].max() < height, f"out of bound event: y = {events['y'].max()}, should less than: height = {height}"

        on_idxs = np.where(events['polarity'] == 1)[0]
        canvas[events['y'][on_idxs], events['x'][on_idxs], :] = [0, 0, 255]
        
        off_idxs = np.where(events['polarity'] == 0)[0]
        canvas[events['y'][off_idxs], events['x'][off_idxs], :] = [255, 0, 0]
    return canvas
'''

def _make_color_img(events, width, height):
    assert events.dtype.names == ('timestamp', 'x', 'y', 'polarity', '_p1', '_p2')
    assert width > 0 and height > 0

    # Last events
    canvas = 255 * np.ones((height, width, 3), dtype=np.uint8)

    if events.size == 0:
        return canvas

    assert events['x'].max() < width, f"out of bound event: x = {events['x'].max()}, should less than: width = {width}"
    assert events['y'].max() < height, f"out of bound event: y = {events['y'].max()}, should less than: height = {height}"

    # 每个像素在canvas的线性索引, x, y都是0基
    idx = events['y'] * width + events['x']

    # 取每个像素最后一次出现的事件
    _, last_pos = np.unique(idx[::-1], return_index=True)  #返回唯一值, 和第一次出现的位置(反转)
    keep = len(idx) - 1 - last_pos  # 反转回原来的索引顺序

    kept_events = events[keep]

    on = kept_events['polarity'] == 1
    off = kept_events['polarity'] == 0

    canvas[kept_events['y'][on], kept_events['x'][on]] = [0, 0, 255]
    canvas[kept_events['y'][off], kept_events['x'][off]] = [255, 0, 0]

    return canvas


def aedat4_to_img_events(aedat_path, event_num, out_dir, width, height):
    """
    Aggregate events from aedat4 to img per fixed events window
    """

    with AedatFile(str(aedat_path)) as f:
        events = np.hstack([event for event in f['events'].numpy()])
        # print(events[0].dtype)  # (timestamp, x, y, polarity, _p1, _p2)
        frame_idx = 0

        output_dir = Path(out_dir)
        output_dir.mkdir(exist_ok=True)

        img_dir = output_dir / 'aedat4_imgs_events'  # 在当前目录创建输出图片的文件夹
        img_dir.mkdir(exist_ok=True)

        # saving the frame_idxs and timestamps for frame images.
        ts_list = []

        for event_idx in tqdm(range(0, len(events), event_num), leave=False, desc='Converting aedat4 to image by fixed events window'):
            rec_events = events[event_idx:event_idx+event_num]  # fixed number of events per image
            event_img = _make_color_img(rec_events, width=width, height=height)
            frame_idx += 1

            img_path = img_dir / f'frame_{frame_idx:06d}.png'
            cv2.imwrite(str(img_path), event_img)
            # cv2.imshow('event_img', event_img)
            # cv2.waitKey(500)

            ts_list.append([frame_idx, events['timestamp'][event_idx]])

        ts_df = pd.DataFrame(ts_list, columns=['frame_idx', 'timestamp'])
        ts_df.to_csv(output_dir / 'aedat4_img_timestamp.csv', index=False)

def aedat4_to_img_time(aedat_path, duration, out_dir, width, height):
    """
    Aggregate events from aedat4 to img per fixed time window (unit=micro seconds)
    """

    with AedatFile(str(aedat_path)) as f:
        events = np.hstack([event for event in f['events'].numpy()])
        # print(events[0].dtype)  # (timestamp, x, y, polarity, _p1, _p2)
        frame_idx = 0

        output_dir = Path(out_dir)
        output_dir.mkdir(exist_ok=True)

        img_dir = output_dir / 'aedat4_imgs_time'  # 在当前目录创建输出图片的文件夹
        img_dir.mkdir(exist_ok=True)

        # saving the frame_idxs and timestamps for frame images.
        ts_list = []

        start_ts = events['timestamp'][0]
        event_list = []
        rec_list = []
        last_start = start_ts
        for event in events:
            # while处理跨多个时间窗口
            while event['timestamp'] >= duration + last_start:  # if timestamp exceeds the boundary, save the current batch
                rec_list.append(np.array(event_list, dtype=events.dtype))
                event_list = []
                last_start += duration
            event_list.append(event)
        if event_list:  # 确保最后一批事件也被添加到 rec_list 中
            rec_list.append(np.array(event_list, dtype=events.dtype))

        for idx, rec_events in enumerate(tqdm(rec_list, leave=False, desc='Converting aedat4 to image by fixed time window')):
            event_img = _make_color_img(rec_events, width=width, height=height)
            frame_idx += 1

            img_path = img_dir / f'frame_{frame_idx:06d}.png'
            cv2.imwrite(str(img_path), event_img)
            # cv2.imshow('event_img', event_img)
            # cv2.waitKey(500)

            ts_list.append([frame_idx, rec_events[0]['timestamp']])

        ts_df = pd.DataFrame(ts_list, columns=['frame_idx', 'timestamp'])
        ts_df.to_csv(output_dir / 'aedat4_img_timestamp.csv', index=False)

def _clip_events(aedat_path, offset):
    """
    Clip events based on the frame timestamps
    The time range is from the start of the first frame to the end of the last frame
    """
    start_frame = offset[aedat_path.parent.name]
    # aedat中的frame比gt对应的frame更多,需要补偿偏移
    frame_dir = aedat_path.parent/'img'
    frame_num = len(list(frame_dir.iterdir()))

    with AedatFile(str(aedat_path)) as f:
        events = np.hstack([event for event in f['events'].numpy()])
        frame_start_list = []
        count = 0
        for frame in f['frames']:
            count += 1
            if start_frame <= count <= start_frame + frame_num:  # 在gt对应范围内,记录每一帧起止时间
                frame_start_list.append(frame.timestamp_start_of_frame)
            else:
                continue
    # clipping
    events = events[events['timestamp'] >= frame_start_list[0]]
    events = events[events['timestamp'] < frame_start_list[-1]]  # 将events流裁切到gt对应的范围内
    # frame_start_list长度是frame+1
    return events, frame_start_list

def aedat4_to_img_frame(aedat_path, offset_path, out_dir, width=346, height=260, sub_div=1):
    """
    Convert aedat4 to images by the original frame timestamps, which means having the same amount as the frame images
    Note: can be subdivided
    """
    offset = {}  # 生成起始帧偏移字典,用于对齐
    with open(offset_path, 'r') as f:
        for line in f.readlines():
            name, start = line.split()
            offset[name] = int(start)+1

    events, frame_start_list = _clip_events(Path(aedat_path), offset)
    frame_start_num= len(frame_start_list)
    sub_frame_start = np.linspace(frame_start_list[0], frame_start_list[1], sub_div+1)  # 对每一帧更进一步切分

    frame_idx = 1
    sub_idx = 1

    output_dir = Path(out_dir)
    output_dir.mkdir(exist_ok=True)

    img_dir = output_dir / 'aedat4_imgs_frame'  # 在当前目录创建输出图片的文件夹
    img_dir.mkdir(exist_ok=True)

    events_list = []

    for event in tqdm(events, leave=False, desc='Converting aedat4 to image by original frame interval'):
        # self._clip_events返回的frame_start_list长度是frame+1
        # 最后一个元素是额外一帧的开始时间,所以不需要处理最后一个start之后的序列, 当frame_idx==frame_num时无需额外处理
        if frame_idx < frame_start_num and event['timestamp'] >= frame_start_list[frame_idx]:  # 如果事件超过当前帧时间
            event_img = _make_color_img(np.array(events_list, dtype=events.dtype), width=width, height=height)
            events_list = [event]
            cv2.imwrite(str(img_dir / f'frame_{((frame_idx - 1) * sub_div + sub_idx):06d}.png'), event_img)
            frame_idx += 1
            sub_frame_start = np.linspace(frame_start_list[frame_idx - 1], frame_start_list[frame_idx], sub_div+1)
            sub_idx = 1
        else:  # 在当前帧时间内
            if  event['timestamp'] >= sub_frame_start[sub_idx]:
                event_img = _make_color_img(np.array(events_list, dtype=events.dtype), width=width, height=height)
                events_list = [event]
                cv2.imwrite(str(img_dir / f'frame_{((frame_idx - 1) * sub_div + sub_idx):06d}.png'), event_img)
                sub_idx += 1
            else:
                events_list.append(event)
    if events_list:
        # 补最后一次
        event_img = _make_color_img(np.array(events_list, dtype=events.dtype), width=width, height=height)
        cv2.imwrite(str(img_dir / f'frame_{((frame_idx - 1) * sub_div + sub_idx):06d}.png'), event_img)

    ts_df = pd.DataFrame({'frame_idx': range(1, len(frame_start_list)),
                          'timestamp': frame_start_list[:-1]})
    ts_df.to_csv(output_dir / 'aedat4_img_frame_timestamp.csv', index=False)


if __name__ == '__main__':
    aedat_path = '/home/yanjiezhang/Downloads/cc/dove/events.aedat4'
    event_num = 10000
    out_dir = '/home/yanjiezhang/Downloads/cc/dove'
    width, height = 346, 260
    offset = '/home/yanjiezhang/Downloads/cc/offset.txt'
    aedat4_to_img_frame(Path(aedat_path), Path(offset), out_dir, width, height, sub_div=2)