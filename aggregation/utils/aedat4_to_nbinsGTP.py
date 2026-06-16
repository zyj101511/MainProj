import os
import cv2
from dv import AedatFile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from aggregation.utils.general import clip_events, get_offset, unique_dir

def _generate_chs(events, last_pos, last_neg, last_hidden, width,
                  height, ch12_strength, ch3_strength, ch3_decay_rate, first_frame=False):
    # 初始化ch1, 2, 3
    pos = np.zeros((height, width))
    neg = np.zeros((height, width))
    hidden = last_hidden * ch3_decay_rate

    # ch1和ch2
    mask_pos = events['polarity'] == 1
    mask_neg = ~mask_pos
    np.add.at(pos, (events['y'][mask_pos], events['x'][mask_pos]), ch12_strength)
    np.add.at(neg, (events['y'][mask_neg], events['x'][mask_neg]), ch12_strength)

    pos = np.clip(pos, 0, 255).astype(np.uint8)
    neg = np.clip(neg, 0, 255).astype(np.uint8)

    # ch3
    if first_frame:
        return pos, neg, hidden
    hidden = _generate_ch3(pos=pos, last_pos=last_pos, neg=neg, last_neg=last_neg, hidden=hidden, ch3_strength=ch3_strength)

    pos = pos + last_pos * 0
    neg = neg + last_neg * 0
    return pos, neg, hidden

def _generate_ch3(pos, last_pos, neg, last_neg, hidden, ch3_strength):
    # pos_map和neg_map代表上一帧中没有事件但是当前帧有事件的位置
    pos_map = (last_pos==0) & (pos != 0)
    neg_map = (last_neg==0) & (neg != 0)

    # 满足上一帧没有事件,当前帧有事件的位置增加strength
    hidden[pos_map] += ch3_strength
    hidden[neg_map] += ch3_strength

    hidden = np.clip(hidden, 0, 255).astype(np.uint8)
    return hidden

def _concat_channels(pos, neg, hidden):
    assert pos.shape == neg.shape == hidden.shape
    height, width = pos.shape
    GTP = np.zeros((3, height, width), dtype=np.uint8)
    GTP[0, :, :] = pos
    GTP[1, :, :] = neg
    GTP[2, :, :] = hidden
    return GTP

def _make_color_GTP(GTP):
    assert GTP.ndim == 3 and GTP.shape[0] == 3
    canvas = GTP.transpose(1, 2, 0)
    return canvas

def aedat4_to_nbinsGTP(aedat_path, offset_path, out_dir, folder_name = 'imgs_GTP', width = 346,
                  height = 260, ch12_strength=40, ch3_strength=30, ch3_decay_rate=0.9, num_bins=4):
    aedat_path = Path(aedat_path)
    offset = get_offset(offset_path)  # 生成起始帧偏移字典,用于对齐

    events, frame_start_list = clip_events(aedat_path, offset)
    frame_start_num= len(frame_start_list)
    sub_frame_start = np.linspace(frame_start_list[0], frame_start_list[1], num_bins+1)  # 对每一帧更进一步切分


    frame_idx = 1
    sub_idx = 1

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = output_dir / folder_name  # 在当前目录创建输出图片的文件夹
    img_dir = unique_dir(img_dir)
    img_dir.mkdir(exist_ok=True, parents=True)

    start_idx = 0  # 窗口的起始event索引

    last_pos = np.zeros((height, width), dtype=np.uint8)
    last_neg= np.zeros((height, width), dtype=np.uint8)
    last_hidden = np.zeros((height, width), dtype=np.uint8)
    first_frame = True  # 第一帧默认ch3全0

    for idx, event in enumerate(tqdm(events, leave=False, desc='Converting aedat4 to GTP')):
        # self._clip_events返回的frame_start_list长度是frame+1
        # 最后一个元素是额外一帧的开始时间,所以不需要处理最后一个start之后的序列, 当frame_idx==frame_num时无需额外处理
        if frame_idx < frame_start_num and event['timestamp'] >= frame_start_list[frame_idx]:  # 如果事件超过当前帧时间
            pos, neg, hidden = _generate_chs(events=events[start_idx: idx], last_pos=last_pos, last_neg=last_neg,
                                             last_hidden=last_hidden, width=width, height=height,
                                             ch12_strength=ch12_strength, ch3_strength=ch3_strength,
                                             ch3_decay_rate=ch3_decay_rate, first_frame=first_frame)
            first_frame = False
            last_pos, last_neg, last_hidden = pos, neg, hidden
            GTP = _concat_channels(pos, neg, hidden)
            GTP_img = _make_color_GTP(GTP)
            start_idx = idx
            cv2.imwrite(str(img_dir / f'frame_{(frame_idx):06d}.png'), GTP_img)
            frame_idx += 1
            sub_frame_start = np.linspace(frame_start_list[frame_idx - 1], frame_start_list[frame_idx], num_bins+1)
            sub_idx = 1
        else:  # 在当前帧时间内
            if  event['timestamp'] >= sub_frame_start[sub_idx]:
                pos, neg, hidden = _generate_chs(events=events[start_idx: idx], last_pos=last_pos, last_neg=last_neg,
                                                 last_hidden=last_hidden, width=width, height=height,
                                                 ch12_strength=ch12_strength, ch3_strength=ch3_strength,
                                                 ch3_decay_rate=ch3_decay_rate, first_frame=first_frame)
                first_frame = False
                last_pos, last_neg, last_hidden = pos, neg, hidden
                start_idx = idx
                sub_idx += 1

    if start_idx < len(events):
        # 补最后一次
        pos, neg, hidden = _generate_chs(events=events[start_idx:], last_pos=last_pos, last_neg=last_neg,
                                         last_hidden=last_hidden, width=width, height=height,
                                         ch12_strength=ch12_strength, ch3_strength=ch3_strength,
                                         ch3_decay_rate=ch3_decay_rate, first_frame=first_frame)
        GTP = _concat_channels(pos, neg, hidden)
        GTP_img = _make_color_GTP(GTP)
        cv2.imwrite(str(img_dir / f'frame_{(frame_idx):06d}.png'), GTP_img)

    # 保存时间戳
    ts_df = pd.DataFrame({'frame_idx': range(1, len(frame_start_list)),
                          'timestamp': frame_start_list[:-1]})
    ts_df.to_csv(output_dir / f'{folder_name}_timestamp.csv', index=False)


if __name__ == '__main__':
    aedat_path = '/home/yanjiezhang/Downloads/cc/dove/events.aedat4'
    out_dir = '/home/yanjiezhang/Downloads/cc/dove'
    width, height = 346, 260
    offset = '/home/yanjiezhang/Downloads/cc/offset.txt'
    aedat4_to_nbinsGTP(Path(aedat_path), Path(offset),
                    out_dir, width=width, height=height, num_bins=4, ch3_strength=30, ch3_decay_rate=0.9)
