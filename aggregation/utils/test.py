import os
import cv2
from dv import AedatFile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from aggregation.utils.general import clip_events, get_offset, unique_dir


def _events_to_bins(events, num_bins):
    """
    把一段事件按时间映射到 [0, num_bins - 1] 的连续坐标上，
    并返回用于线性插值的左/右 bin 索引和权重。
    """
    if len(events) == 0:
        return None

    events = events.copy()

    first_timestamp = events['timestamp'][0]
    last_timestamp = events['timestamp'][-1]
    delta_ts = last_timestamp - first_timestamp
    if delta_ts == 0:
        delta_ts = 1.0

    t = (events['timestamp'] - first_timestamp) / delta_ts
    t = t * (num_bins - 1)

    t_left = np.floor(t).astype(np.int64)
    t_right = t_left + 1
    w_right = t - t_left
    w_left = 1.0 - w_right

    valid_left = (t_left >= 0) & (t_left < num_bins)
    valid_right = (t_right >= 0) & (t_right < num_bins)

    return t_left, t_right, w_left, w_right, valid_left, valid_right


def _accumulate_interp(volume, events, num_bins, strength=1.0):
    """
    类似 voxel 的线性插值，把事件累积到相邻两个时间 bin。
    volume: (num_bins, H, W)
    """
    if len(events) == 0:
        return volume

    t_left, t_right, w_left, w_right, valid_left, valid_right = _events_to_bins(events, num_bins)

    pol = events['polarity'].copy()
    pol[pol == 0] = -1
    pol = pol.astype(np.float32)

    y = events['y']
    x = events['x']

    # 左 bin
    np.add.at(
        volume,
        (t_left[valid_left], y[valid_left], x[valid_left]),
        strength * pol[valid_left] * w_left[valid_left]
    )

    # 右 bin
    np.add.at(
        volume,
        (t_right[valid_right], y[valid_right], x[valid_right]),
        strength * pol[valid_right] * w_right[valid_right]
    )

    return volume


def _generate_chs_voxel_style(events, width, height, num_bins,
                              ch12_strength=40, ch3_strength=30, ch3_decay_rate=0.9,
                              last_hidden=None):
    """
    生成 voxel-style 的三通道 GTP：
    - ch1: 正极性事件，带时间插值
    - ch2: 负极性事件，带时间插值
    - ch3: 基于历史 hidden 衰减 + 当前事件增量，且当前事件也按时间插值
    """
    pos = np.zeros((num_bins, height, width), dtype=np.float32)
    neg = np.zeros((num_bins, height, width), dtype=np.float32)

    if last_hidden is None:
        hidden = np.zeros((num_bins, height, width), dtype=np.float32)
    else:
        hidden = last_hidden.astype(np.float32) * ch3_decay_rate

    if len(events) == 0:
        return pos, neg, hidden

    # 时间插值
    t_left, t_right, w_left, w_right, valid_left, valid_right = _events_to_bins(events, num_bins)

    y = events['y']
    x = events['x']

    mask_pos = events['polarity'] == 1
    mask_neg = ~mask_pos

    # ch1: 正极性，按时间插值
    if np.any(mask_pos):
        pos_events = events[mask_pos]
        pos_t_left, pos_t_right, pos_w_left, pos_w_right, pos_valid_left, pos_valid_right = _events_to_bins(pos_events,
                                                                                                            num_bins)
        py = pos_events['y']
        px = pos_events['x']

        np.add.at(
            pos,
            (pos_t_left[pos_valid_left], py[pos_valid_left], px[pos_valid_left]),
            ch12_strength * pos_w_left[pos_valid_left]
        )
        np.add.at(
            pos,
            (pos_t_right[pos_valid_right], py[pos_valid_right], px[pos_valid_right]),
            ch12_strength * pos_w_right[pos_valid_right]
        )

    # ch2: 负极性，按时间插值
    if np.any(mask_neg):
        neg_events = events[mask_neg]
        neg_t_left, neg_t_right, neg_w_left, neg_w_right, neg_valid_left, neg_valid_right = _events_to_bins(neg_events,
                                                                                                            num_bins)
        ny = neg_events['y']
        nx = neg_events['x']

        np.add.at(
            neg,
            (neg_t_left[neg_valid_left], ny[neg_valid_left], nx[neg_valid_left]),
            ch12_strength * neg_w_left[neg_valid_left]
        )
        np.add.at(
            neg,
            (neg_t_right[neg_valid_right], ny[neg_valid_right], nx[neg_valid_right]),
            ch12_strength * neg_w_right[neg_valid_right]
        )

    # ch3: 基于 ch1/ch2 的“新变化位置”生成，再按时间插值注入 hidden
    if last_hidden is not None:
        # 先做一个当前帧的“变化图”
        current_pos = pos.sum(axis=0)
        current_neg = neg.sum(axis=0)

        last_pos = np.zeros_like(current_pos)
        last_neg = np.zeros_like(current_neg)

        # 这里用历史 hidden 的状态，只保留衰减
        hidden = hidden

        # 找当前窗中有事件、上一状态无事件的位置
        pos_map = (last_pos == 0) & (current_pos != 0)
        neg_map = (last_neg == 0) & (current_neg != 0)

        # 把这些位置均匀注入到整个时间轴对应的位置
        # 这里为了保持 voxel 风格，按每个事件的时间位置注入，而不是只注入到空间位置一次
        # 所以用事件级别再做一遍累积更合理
        if len(events) > 0:
            hidden = _accumulate_ch3_interp(
                hidden, events, num_bins, ch3_strength=ch3_strength
            )

    return np.clip(pos, 0, 255).astype(np.uint8), np.clip(neg, 0, 255).astype(np.uint8), np.clip(hidden, 0, 255).astype(
        np.uint8)


def _accumulate_ch3_interp(hidden, events, num_bins, ch3_strength=30):
    """
    ch3 的 voxel-style 插值版本：
    - 保留 hidden 本身
    - 对当前事件按时间分到相邻 bins
    - 由于 ch3 是“历史增强通道”，这里用绝对增量方式注入
    """
    if len(events) == 0:
        return hidden

    t_left, t_right, w_left, w_right, valid_left, valid_right = _events_to_bins(events, num_bins)

    # ch3 不直接区分极性也可以；这里保留极性信息作为增量方向
    pol = events['polarity'].copy()
    pol[pol == 0] = -1
    pol = pol.astype(np.float32)

    y = events['y']
    x = events['x']

    # 左 bin
    np.add.at(
        hidden,
        (t_left[valid_left], y[valid_left], x[valid_left]),
        ch3_strength * np.abs(pol[valid_left]) * w_left[valid_left]
    )

    # 右 bin
    np.add.at(
        hidden,
        (t_right[valid_right], y[valid_right], x[valid_right]),
        ch3_strength * np.abs(pol[valid_right]) * w_right[valid_right]
    )

    return hidden


def _concat_channels(pos, neg, hidden):
    """
    输入:
      pos, neg, hidden: (num_bins, H, W)
    输出:
      GTP: (3, num_bins, H, W)
    """
    assert pos.shape == neg.shape == hidden.shape
    num_bins, height, width = pos.shape
    GTP = np.zeros((3, num_bins, height, width), dtype=np.uint8)
    GTP[0, :, :, :] = pos
    GTP[1, :, :, :] = neg
    GTP[2, :, :, :] = hidden
    return GTP


def _make_color_GTP(GTP, mode='sum'):
    """
    把 3D/4D GTP 压成可视化图。
    输入 GTP:
      - (3, num_bins, H, W)
    输出:
      - (H, W, 3)
    """
    assert GTP.ndim == 4 and GTP.shape[0] == 3
    if mode == 'sum':
        canvas = GTP.sum(axis=1)  # (3, H, W)
    elif mode == 'max':
        canvas = GTP.max(axis=1)
    elif mode == 'mean':
        canvas = GTP.mean(axis=1)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    canvas = canvas.transpose(1, 2, 0)  # (H, W, 3)
    return canvas.astype(np.uint8)


def aedat4_to_nbinsGTP(aedat_path, offset_path, out_dir, folder_name='imgs_GTP',
                       width=346, height=260, ch12_strength=40, ch3_strength=30,
                       ch3_decay_rate=0.9, num_bins=4, vis_mode='sum'):
    aedat_path = Path(aedat_path)
    offset = get_offset(offset_path)

    events, frame_start_list = clip_events(aedat_path, offset)
    frame_start_num = len(frame_start_list)
    sub_frame_start = np.linspace(frame_start_list[0], frame_start_list[1], num_bins + 1)

    frame_idx = 1
    sub_idx = 1

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = output_dir / folder_name
    img_dir = unique_dir(img_dir)
    img_dir.mkdir(exist_ok=True, parents=True)

    start_idx = 0

    last_hidden = np.zeros((num_bins, height, width), dtype=np.float32)
    first_frame = True

    for idx, event in enumerate(tqdm(events, leave=False, desc='Converting aedat4 to nbins GTP')):
        if frame_idx < frame_start_num and event['timestamp'] >= frame_start_list[frame_idx]:
            pos, neg, hidden = _generate_chs_voxel_style(
                events=events[start_idx:idx],
                width=width,
                height=height,
                num_bins=num_bins,
                ch12_strength=ch12_strength,
                ch3_strength=ch3_strength,
                ch3_decay_rate=ch3_decay_rate,
                last_hidden=last_hidden
            )

            first_frame = False
            last_hidden = hidden.astype(np.float32)

            GTP = _concat_channels(pos, neg, hidden)
            GTP_img = _make_color_GTP(GTP, mode=vis_mode)

            start_idx = idx
            cv2.imwrite(str(img_dir / f'frame_{frame_idx:06d}.png'), GTP_img)

            frame_idx += 1
            if frame_idx < frame_start_num:
                sub_frame_start = np.linspace(frame_start_list[frame_idx - 1], frame_start_list[frame_idx],
                                              num_bins + 1)
            sub_idx = 1

        else:
            if sub_idx < num_bins and event['timestamp'] >= sub_frame_start[sub_idx]:
                pos, neg, hidden = _generate_chs_voxel_style(
                    events=events[start_idx:idx],
                    width=width,
                    height=height,
                    num_bins=num_bins,
                    ch12_strength=ch12_strength,
                    ch3_strength=ch3_strength,
                    ch3_decay_rate=ch3_decay_rate,
                    last_hidden=last_hidden
                )

                first_frame = False
                last_hidden = hidden.astype(np.float32)
                start_idx = idx
                sub_idx += 1

    if start_idx < len(events):
        pos, neg, hidden = _generate_chs_voxel_style(
            events=events[start_idx:],
            width=width,
            height=height,
            num_bins=num_bins,
            ch12_strength=ch12_strength,
            ch3_strength=ch3_strength,
            ch3_decay_rate=ch3_decay_rate,
            last_hidden=last_hidden
        )

        GTP = _concat_channels(pos, neg, hidden)
        GTP_img = _make_color_GTP(GTP, mode=vis_mode)
        cv2.imwrite(str(img_dir / f'frame_{frame_idx:06d}.png'), GTP_img)

    ts_df = pd.DataFrame({
        'frame_idx': range(1, len(frame_start_list)),
        'timestamp': frame_start_list[:-1]
    })
    ts_df.to_csv(output_dir / f'{folder_name}_timestamp.csv', index=False)


if __name__ == '__main__':
    aedat_path = '/home/yanjiezhang/Downloads/cc/dove/events.aedat4'
    out_dir = '/home/yanjiezhang/Downloads/cc/dove'
    width, height = 346, 260
    offset = '/home/yanjiezhang/Downloads/cc/offset.txt'

    aedat4_to_nbinsGTP(
        Path(aedat_path),
        Path(offset),
        out_dir,
        width=width,
        height=height,
        num_bins=4,
        ch12_strength=40,
        ch3_strength=30,
        ch3_decay_rate=0.8,
        vis_mode='sum'
    )