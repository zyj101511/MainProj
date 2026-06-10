import os
import sys
from os.path import join

import cv2
import numpy as np
from dv import AedatFile
from tqdm import tqdm
import shutil
import argparse

os.sep = '/'

np.set_printoptions(suppress=True)

pair = {}


def get_start_frame(seq_name):
    return pair[seq_name]


def trans_pair(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    source_file = os.path.join(source_dir, 'pair.txt')
    target_file = os.path.join(target_dir, 'pair.txt')

    if os.path.isfile(source_file):
        shutil.copy(source_file, target_file)
        print(f"Copied {source_file} to {target_file}")
    else:
        print(f"Source file {source_file} does not exist.")


def folder_trans(source_dir, target_dir):
    for split_dir in os.listdir(source_dir):
        if split_dir in ["train", "test"]:
            split_source_dir = os.path.join(source_dir, split_dir)
            split_target_dir = os.path.join(target_dir, split_dir)

            if not os.path.exists(split_target_dir):
                os.makedirs(split_target_dir)

            for sub_dir in os.listdir(split_source_dir):
                if sub_dir in ["test.txt", "train.txt"]:
                    continue
                if sub_dir != "inter3_stack":
                    sub_source_dir = os.path.join(split_source_dir, sub_dir)
                    sub_target_dir = os.path.join(split_target_dir, sub_dir)

                    shutil.copytree(
                        sub_source_dir,
                        sub_target_dir,
                        ignore=shutil.ignore_patterns('inter3_stack')
                    )
                    print(f"复制 {sub_dir} 完成")
    print("数据集复制完成")


def text_generate(target_dir):
    text_root = os.path.join(target_dir, "train")
    folder_names = [name for name in os.listdir(text_root) if os.path.isdir(os.path.join(text_root, name))]
    output_file = os.path.join(text_root, "test.txt")
    with open(output_file, 'w') as file:
        for name in folder_names:
            file.write(name + '\n')
    print("文件名已写入到", output_file)

    text_root = os.path.join(target_dir, "test")
    folder_names = [name for name in os.listdir(text_root) if os.path.isdir(os.path.join(text_root, name))]
    output_file = os.path.join(text_root, "test.txt")
    with open(output_file, 'w') as file:
        for name in folder_names:
            file.write(name + '\n')
    print("文件名已写入到", output_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for generate dataset')
    parser.add_argument('--trans_folder', type=int, default=0, help='Set to 1 if dataset copying is needed')
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Only required when trans_folder is True')
    parser.add_argument('--target_dir', type=str, default=None, help='Target dataset path')
    parser.add_argument('--stack_name', type=str, default='inter3_stack', help='name of the stack file')

    parser.add_argument('--s_train', type=int, default=0, help='start index for train dataset')
    parser.add_argument('--e_train', type=int, default=0, help='end index for train dataset')
    parser.add_argument('--s_test', type=int, default=0, help='start index for test dataset')
    parser.add_argument('--e_test', type=int, default=0, help='end index for test dataset')

    parser.add_argument('--stack_amount_1c2c', type=float, default=40,
                        help='Value of single stacking for the first two channels')
    parser.add_argument('--stack_amount_3c', type=float, default=64,
                        help='Value of single stacking for the third channel (hidden_pic)')
    parser.add_argument('--decay_rate_3c', type=float, default=0.8,
                        help='Decay of values between sub-frames for the third channel')
    parser.add_argument('--sub_div', type=int, default=4,
                        help='Number of internal time slices inside one original frame interval')

    args = parser.parse_args()
    return args


def save_3c_img(pos_img, neg_img, hidden_img, root):
    img = np.zeros((pos_img.shape[0], pos_img.shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = pos_img
    img[:, :, 1] = neg_img
    img[:, :, 2] = hidden_img
    cv2.imwrite(root, img)


def process_event(pos_img, neg_img, event, pic_shape, stack_amount_1c2c):
    x, y, p = int(event[1]), int(event[2]), int(event[3])
    if 0 <= x < pic_shape[1] and 0 <= y < pic_shape[0]:
        if p == 1:
            pos_img[y, x] = min(255, pos_img[y, x] + stack_amount_1c2c)
        else:
            neg_img[y, x] = min(255, neg_img[y, x] + stack_amount_1c2c)


def update_hidden(hidden_img, pos_img, neg_img, last_pos_img, last_neg_img, stack_amount_3c, decay_rate_3c):
    """
    让新内容更“深”，旧内容逐步衰减。
    这里不是做简单差分，而是：
    - 先衰减历史 hidden
    - 当前子切片中新出现的位置加分
    """
    new_hidden = hidden_img.astype(np.float32) * decay_rate_3c

    pos_new = (last_pos_img == 0) & (pos_img != 0)
    neg_new = (last_neg_img == 0) & (neg_img != 0)

    new_hidden[pos_new] += stack_amount_3c
    new_hidden[neg_new] += stack_amount_3c

    new_hidden = np.clip(new_hidden, 0, 255).astype(np.uint8)
    return new_hidden


def stack_event(stack_name, index, root, pair_root, stack_amount_1c2c, stack_amount_3c,
                decay_rate_3c, sub_div=4):
    match_file = pair_root
    with open(match_file, 'r') as f:
        for line in f.readlines():
            file, start_frame = line.split()
            pair[file] = int(start_frame) + 1

    seq_name = root.split('/')[-1]

    img_path = os.path.join(root, 'img').replace('\\', '/')
    frame_num = len(os.listdir(img_path))
    event_data = os.path.join(root, 'events.aedat4').replace('\\', '/')
    stack_path = os.path.join(root, stack_name).replace('\\', '/')

    start_frame = get_start_frame(seq_name)

    if not os.path.exists(stack_path):
        os.mkdir(stack_path)

    with AedatFile(event_data) as f:
        pic_shape = f['events'].size
        events = np.hstack([packet for packet in f['events'].numpy()])

        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        event = np.vstack((timestamps, x, y, polarities))
        event = np.swapaxes(event, 0, 1)

        time_series = []
        count = 0
        for frame in f["frames"]:
            count += 1
            if count >= start_frame and count <= start_frame + frame_num:
                time_series.append(frame.timestamp_start_of_frame)

        event = event[event[:, 0] >= time_series[0]]
        event = event[event[:, 0] < time_series[-1]]

        deal_event(
            index=index,
            events=event,
            frame_timestamp=time_series,
            pic_shape=pic_shape,
            save_name=stack_path,
            stack_amount_1c2c=stack_amount_1c2c,
            stack_amount_3c=stack_amount_3c,
            decay_rate_3c=decay_rate_3c,
            sub_div=sub_div
        )


def deal_event(index, events, frame_timestamp, pic_shape, save_name,
               stack_amount_1c2c, stack_amount_3c, decay_rate_3c, sub_div=4):
    """
    每个原始 frame 区间只输出 1 张图。
    区间内部再细分成 sub_div 份，但不单独保存。
    """
    if len(frame_timestamp) < 2:
        return

    frame_id = 1  # 当前原始 frame 区间编号
    sub_id = 0

    pos_img = np.zeros(pic_shape, dtype=np.uint8)
    neg_img = np.zeros(pic_shape, dtype=np.uint8)
    hidden_state = np.zeros(pic_shape, dtype=np.uint8)

    last_pos_img = np.zeros(pic_shape, dtype=np.uint8)
    last_neg_img = np.zeros(pic_shape, dtype=np.uint8)

    # 当前大窗内的子窗边界
    sub_bounds = np.linspace(frame_timestamp[0], frame_timestamp[1], sub_div + 1)
    current_sub_end = sub_bounds[sub_id + 1]

    for event in tqdm(events, desc=f"{index} Writing {save_name.split('/')[-1]} events"):
        t = event[0]

        # 如果跨过当前原始 frame 的结束边界，先把当前大窗内剩余内容保存掉
        while frame_id < len(frame_timestamp) - 1 and t >= frame_timestamp[frame_id]:
            # 保存当前原始 frame 的最终结果
            out_path = save_name + '/' + str(frame_id).zfill(4) + '.png'
            save_3c_img(pos_img, neg_img, hidden_state, out_path)

            # 进入下一原始 frame 区间
            frame_id += 1
            if frame_id >= len(frame_timestamp):
                break

            # 重置当前大窗内状态
            pos_img = np.zeros(pic_shape, dtype=np.uint8)
            neg_img = np.zeros(pic_shape, dtype=np.uint8)
            hidden_state = np.zeros(pic_shape, dtype=np.uint8)
            last_pos_img = np.zeros(pic_shape, dtype=np.uint8)
            last_neg_img = np.zeros(pic_shape, dtype=np.uint8)
            sub_id = 0

            if frame_id < len(frame_timestamp):
                sub_bounds = np.linspace(frame_timestamp[frame_id - 1], frame_timestamp[frame_id], sub_div + 1)
                current_sub_end = sub_bounds[1]

        if frame_id >= len(frame_timestamp):
            break

        # 当前事件落在当前原始 frame 区间内
        # 如果越过了当前子切片边界，先更新 hidden，再进入下一子切片
        while t >= current_sub_end and sub_id < sub_div - 1:
            hidden_state = update_hidden(
                hidden_state,
                pos_img,
                neg_img,
                last_pos_img,
                last_neg_img,
                stack_amount_3c,
                decay_rate_3c
            )
            last_pos_img = pos_img.copy()
            last_neg_img = neg_img.copy()
            pos_img = np.zeros(pic_shape, dtype=np.uint8)
            neg_img = np.zeros(pic_shape, dtype=np.uint8)

            sub_id += 1
            current_sub_end = sub_bounds[sub_id + 1]

        process_event(pos_img, neg_img, event, pic_shape, stack_amount_1c2c)

    # 保存最后一个原始 frame 区间
    if frame_id < len(frame_timestamp):
        hidden_state = update_hidden(
            hidden_state,
            pos_img,
            neg_img,
            last_pos_img,
            last_neg_img,
            stack_amount_3c,
            decay_rate_3c
        )
        out_path = save_name + '/' + str(frame_id).zfill(4) + '.png'
        save_3c_img(pos_img, neg_img, hidden_state, out_path)


def stack_dataset(root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c, sub_div=4):
    file_name_list = []
    text_root = os.path.join(root, "test.txt")
    pair_root = os.path.join(target_dir, "pair.txt")

    with open(text_root, 'r') as f:
        a = [i.strip() for i in f.readlines()]
        for line in a:
            file_name_list.append(line)

    for index, i in enumerate(sorted(file_name_list)[:]):
        data = os.path.join(root, i).replace('\\', '/')
        if os.path.exists(join(data, stack_name).replace('\\', '/')):
            if 3 * len(os.listdir(join(data, 'img').replace('\\', '/'))) == len(
                    os.listdir(join(data, stack_name).replace('\\', '/'))):
                continue
        stack_event(
            stack_name, index, data, pair_root,
            stack_amount_1c2c, stack_amount_3c, decay_rate_3c,
            sub_div=sub_div
        )


args = parse_args()
trans_folder = args.trans_folder
source_dir = args.source_dir
target_dir = args.target_dir
stack_name = args.stack_name

s_train = args.s_train
e_train = args.e_train
s_test = args.s_test
e_test = args.e_test

stack_amount_1c2c = args.stack_amount_1c2c
stack_amount_3c = args.stack_amount_3c
decay_rate_3c = args.decay_rate_3c
sub_div = args.sub_div

train_root = os.path.join(target_dir, "train")
test_root = os.path.join(target_dir, "test")

if trans_folder == 1:
    if source_dir is None:
        print("Missing original dataset path")
        sys.exit()
    trans_pair(source_dir, target_dir)
    folder_trans(source_dir, target_dir)
    text_generate(target_dir)

stack_dataset(train_root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c, sub_div=sub_div)
stack_dataset(test_root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c, sub_div=sub_div)