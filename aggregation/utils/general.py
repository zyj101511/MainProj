from pathlib import Path
from dv import AedatFile
import numpy as np


def unique_dir(dir):
    dir = Path(dir)
    if not dir.exists():
        return dir
    parent = dir.parent
    name = dir.name
    idx = 1
    while True:
        candidate = parent / f'{name}_{idx:06d}'
        if not candidate.exists():
            return candidate
        idx += 1

def get_offset(path):
    offset_dict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            file, offset_frame = line.split()
            offset_dict[file] = int(offset_frame) + 1
    return offset_dict

def clip_events(aedat_path, offset):
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