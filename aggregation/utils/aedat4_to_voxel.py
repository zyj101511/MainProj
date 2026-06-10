import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from aggregation.utils.aedat4_to_img import _clip_events


def _make_voxel_img(voxel, width, height, mode='abs_sum'):
    assert width > 0 and height > 0
    # (C, H, W) -> (H, W)
    if mode == 'mean':
        voxel_tmp = voxel.mean(axis=0)
    elif mode == 'max':
        voxel_tmp = voxel.max(axis=0)
    elif mode == 'sum':
        voxel_tmp = voxel.sum(axis=0)
    elif mode == 'abs_sum':
        voxel_tmp = abs(voxel).sum(axis=0)

    vmin = np.percentile(voxel_tmp, 0)
    vmax = np.percentile(voxel_tmp, 99.9)
    voxel_tmp = np.clip(voxel_tmp, vmin, vmax)
    canvas = (voxel_tmp - vmin) / (vmax - vmin + 1e-8)
    canvas = (canvas * 255).astype(np.uint8)

    return canvas

def _window_to_voxel(events, num_bins=1, width=346, height=260):
    """
    Transform a window of events into an n-bins voxel grid representation.
    每个事件都会分到两个相邻 bin 里，不是只进一个 bin。
    """
    assert events.dtype.names == ('timestamp', 'x', 'y', 'polarity', '_p1', '_p2')
    assert num_bins >= 1
    assert width > 0 and height > 0
    events = events.copy()

    # voxel grid容器
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)

    last_timestamp = events['timestamp'][-1]
    first_timestamp = events['timestamp'][0]
    delta_ts = last_timestamp - first_timestamp
    # delta_ts用于归一化
    if delta_ts == 0:
        delta_ts = 1.0  # 处理极端情况如果所有事件同时间,防止溢出

    events['timestamp'] = (events['timestamp'] - first_timestamp) / delta_ts
    events['timestamp'] = (num_bins - 1) * events['timestamp']  # 把每个events的ts映射到0~num_bins-1

    events['polarity'][events['polarity'] == 0] = -1  # off -> -1, on -> 1

    # 取整数部分和小数部分
    ts_integer= events['timestamp'].astype(np.int64)
    ts_decimal = events['timestamp'] - ts_integer

    # 平滑插值, 将事件分到左右两个bin中, 根据decimal大小决定放在两侧的强度
    val_left = events['polarity'] * (1.0 - ts_decimal)
    val_right = events['polarity'] * ts_decimal

    # 把每个事件放入自己左侧的bin
    valid_idxs = ts_integer < num_bins  # 放置浮点计算导致越界, 得到合法的events索引
    np.add.at(voxel, (ts_integer[valid_idxs],
                      events['y'][valid_idxs],
                      events['x'][valid_idxs]),
              val_left[valid_idxs])

    valid_idxs = (ts_integer + 1) < num_bins
    np.add.at(voxel, (ts_integer[valid_idxs]+1,
                      events['y'][valid_idxs],
                      events['x'][valid_idxs]),
              val_right[valid_idxs])

    return voxel  # (num_bin, H, W)

def aedat4_to_voxel(aedat_path, offset_path, out_dir, width=346, height=260, num_bins=1, img_mode='abs_sum'):
    """
    Convert an aedat4 file to a voxel grid. Each voxel grid are
    from a window of events which length is divided by original frame images
    """
    offset = {}  # 生成起始帧偏移字典,用于对齐
    with open(offset_path, 'r') as f:
        for line in f.readlines():
            name, start = line.split()
            offset[name] = int(start)+1

    events, frame_start_list = _clip_events(Path(aedat_path), offset)
    frame_start_num= len(frame_start_list)

    frame_idx = 1

    output_dir = Path(out_dir)
    output_dir.mkdir(exist_ok=True)

    img_dir = output_dir / 'aedat4_voxel'  # 在当前目录创建输出图片的文件夹
    img_dir.mkdir(exist_ok=True)

    events_list = []

    for event in tqdm(events, leave=False, desc='Converting aedat4 to voxel by original frame interval'):
        # self._clip_events返回的frame_start_list长度是frame+1
        # 最后一个元素是额外一帧的开始时间,所以不需要处理最后一个start之后的序列, 当frame_idx==frame_num时无需额外处理
        if frame_idx < frame_start_num and event['timestamp'] >= frame_start_list[frame_idx]:
            voxel = _window_to_voxel(np.array(events_list, dtype=events.dtype),
                                     num_bins=num_bins, width=width, height=height)
            voxel_img = _make_voxel_img(voxel, width=width, height=height, mode=img_mode)
            events_list = [event]
            cv2.imwrite(str(img_dir / f'voxel_{frame_idx:06d}.png'), voxel_img)
            frame_idx += 1
        else:
            events_list.append(event)
    if events_list:
        voxel = _window_to_voxel(np.array(events_list, dtype=events.dtype),
                                 num_bins=num_bins, width=width, height=height)
        voxel_img = _make_voxel_img(voxel, width=width, height=height, mode=img_mode)
        cv2.imwrite(str(img_dir / f'voxel_{frame_idx:06d}.png'), voxel_img)

    ts_df = pd.DataFrame({'voxel_idx': range(1, len(frame_start_list)),
                          'timestamp': frame_start_list[:-1]})
    ts_df.to_csv(output_dir / 'aedat4_voxel_timestamp.csv', index=False)

if __name__ == '__main__':
    aedat_path = '/home/yanjiezhang/Downloads/cc/dove/events.aedat4'
    out_dir = '/home/yanjiezhang/Downloads/cc/dove'
    width, height = 346, 260
    offset = '/home/yanjiezhang/Downloads/cc/offset.txt'
    aedat4_to_voxel(Path(aedat_path), Path(offset),
                    out_dir, width, height,
                    num_bins=2,
                    img_mode='abs_sum')