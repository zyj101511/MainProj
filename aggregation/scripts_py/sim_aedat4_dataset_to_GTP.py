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


def process_event(pos_img, neg_img, event, pic_shape, stack_amount_1c2c):
    x, y, p = int(event[1]), int(event[2]), int(event[3])
    if 0 <= x < pic_shape[1] and 0 <= y < pic_shape[0]:
        if p == 1:
            pos_img[y][x] = min(255, pos_img[y][x] + stack_amount_1c2c)
        else:
            neg_img[y][x] = min(255, neg_img[y][x] + stack_amount_1c2c)


def save_2C_img(pos_img, neg_img, null_img, root):
    two_channel_img = np.zeros((pos_img.shape[0], pos_img.shape[1], 3), dtype=np.uint8)
    two_channel_img[:, :, 0] = pos_img
    two_channel_img[:, :, 1] = neg_img
    two_channel_img[:, :, 2] = null_img
    cv2.imwrite(root, two_channel_img)


def hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_pic, stack_amount_3c, decay_rate_3c):
    new_hidden_pic = hidden_pic.astype(np.float32) * decay_rate_3c

    pos_condition = (last_pos_pic == 0) & (pos_img != 0)
    neg_condition = (last_neg_pic == 0) & (neg_img != 0)

    new_hidden_pic[pos_condition] += stack_amount_3c
    new_hidden_pic[neg_condition] += stack_amount_3c

    new_hidden_pic = np.clip(new_hidden_pic, 0, 255).astype(np.uint8)
    return new_hidden_pic


def deal_event_no_frames(index, events, pic_shape, save_name,
                         stack_amount_1c2c, stack_amount_3c, decay_rate_3c,
                         num_outputs):
    """
    不依赖 frames，直接把事件按时间均匀切成 num_outputs 段。
    每一段输出一张 2C + hidden 图。
    """
    if len(events) == 0:
        print(f"[skip] empty events: {save_name}")
        return

    os.makedirs(save_name, exist_ok=True)

    t0 = events[0, 0]
    t1 = events[-1, 0]
    if t1 <= t0:
        print(f"[skip] invalid time range: {save_name}")
        return

    boundaries = np.linspace(t0, t1, num_outputs + 1)

    last_pos_pic = np.full(pic_shape, 0, dtype=np.uint8)
    last_neg_pic = np.full(pic_shape, 0, dtype=np.uint8)
    hidden_state = np.full(pic_shape, 0, dtype=np.uint8)

    seg_idx = 0
    pos_img = np.full(pic_shape, 0, dtype=np.uint8)
    neg_img = np.full(pic_shape, 0, dtype=np.uint8)
    current_boundary = boundaries[1]

    for event in tqdm(events, desc=f"{index} Writing {os.path.basename(save_name)} events"):
        while event[0] >= current_boundary and seg_idx < num_outputs - 1:
            img_save_root = save_name + '/' + str(seg_idx + 1).zfill(4) + '_1.png'
            if seg_idx > 0:
                hidden_state = hidden_pic_generator(
                    pos_img, neg_img, last_pos_pic, last_neg_pic,
                    hidden_state, stack_amount_3c, decay_rate_3c
                )
            save_2C_img(pos_img, neg_img, hidden_state, img_save_root)

            last_pos_pic = pos_img
            last_neg_pic = neg_img
            pos_img = np.full(pic_shape, 0, dtype=np.uint8)
            neg_img = np.full(pic_shape, 0, dtype=np.uint8)

            seg_idx += 1
            current_boundary = boundaries[min(seg_idx + 1, len(boundaries) - 1)]

        process_event(pos_img, neg_img, event, pic_shape, stack_amount_1c2c)

    img_save_root = save_name + '/' + str(seg_idx + 1).zfill(4) + '_1.png'
    hidden_state = hidden_pic_generator(
        pos_img, neg_img, last_pos_pic, last_neg_pic,
        hidden_state, stack_amount_3c, decay_rate_3c
    )
    save_2C_img(pos_img, neg_img, hidden_state, img_save_root)


def stack_event(stack_name, index, root, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    seq_name = root.split('/')[-1]

    img_path = os.path.join(root, 'img').replace('\\', '/')
    if not os.path.exists(img_path):
        print(f"[skip] missing img folder: {img_path}")
        return

    frame_num = len(os.listdir(img_path))
    event_data = os.path.join(root, 'events.aedat4').replace('\\', '/')
    stack_path = os.path.join(root, stack_name).replace('\\', '/')

    if not os.path.exists(event_data):
        print(f"[skip] missing event file: {event_data}")
        return

    if not os.path.exists(stack_path):
        os.mkdir(stack_path)

    with AedatFile(event_data) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
        timestamps = events['timestamp']
        x = events['x']
        y = events['y']
        polarities = events['polarity']

        event = np.vstack((timestamps, x, y, polarities))
        event = np.swapaxes(event, 0, 1)

        # 输出数量按图片数来定；如果你想更密，可以改成 frame_num * 2 / 3 / 4
        num_outputs = frame_num

        deal_event_no_frames(
            index=index,
            events=event,
            pic_shape=f['events'].size,
            save_name=stack_path,
            stack_amount_1c2c=stack_amount_1c2c,
            stack_amount_3c=stack_amount_3c,
            decay_rate_3c=decay_rate_3c,
            num_outputs=num_outputs
        )


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

                    shutil.copytree(sub_source_dir, sub_target_dir, ignore=shutil.ignore_patterns('inter3_stack'))
                    print(f"复制 {sub_dir} 完成")
    print("数据集复制完成")


def text_generate(target_dir):
    text_root = os.path.join(target_dir, "train")
    folder_names = [name for name in os.listdir(text_root) if os.path.isdir(os.path.join(text_root, name))]
    output_file = os.path.join(text_root, "test.txt")
    with open(output_file, 'w') as file:
        for name in folder_names:
            file.write(name + '\n')
    print("文件夹名已写入到", output_file, "文件中。")

    text_root = os.path.join(target_dir, "test")
    folder_names = [name for name in os.listdir(text_root) if os.path.isdir(os.path.join(text_root, name))]
    output_file = os.path.join(text_root, "test.txt")
    with open(output_file, 'w') as file:
        for name in folder_names:
            file.write(name + '\n')
    print("文件夹名已写入到", output_file, "文件中。")


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for gengrate_dataset')
    parser.add_argument('--trans_folder', type=int, default=0, help='Set to 1 if dataset copying is needed')
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Only required when trans_folder is True, set to the path of the original dataset')
    parser.add_argument('--target_dir', type=str, default=None, help='Target dataset path')
    parser.add_argument('--stack_name', type=str, default='inter3_stack', help='name of the stack file')

    parser.add_argument('--s_train', type=int, default=0, help='start index for train dataset')
    parser.add_argument('--e_train', type=int, default=0, help='start index for train dataset')
    parser.add_argument('--s_test', type=int, default=0, help='start index for test dataset')
    parser.add_argument('--e_test', type=int, default=0, help='start index for test dataset')

    parser.add_argument('--stack_amount_1c2c', type=float, default=40,
                        help='Value of single stacking for the first two channels')
    parser.add_argument('--stack_amount_3c', type=float, default=64,
                        help='Value of single stacking for the third channel (hidden_pic)')
    parser.add_argument('--decay_rate_3c', type=float, default=0.8,
                        help='Decay of values between two frames for the third channel (hidden_pic)')

    args = parser.parse_args()
    return args


def stack_dataset(root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    file_name_list = []
    text_root = os.path.join(root, "test.txt")

    if not os.path.exists(text_root):
        print(f"[skip] missing {text_root}")
        return

    with open(text_root, 'r') as f:
        a = [i.strip() for i in f.readlines()]
        for line in a:
            file_name_list.append(line)

    for index, i in enumerate(sorted(file_name_list)[:]):
        data = os.path.join(root, i).replace('\\', '/')
        stack_dir = join(data, stack_name).replace('\\', '/')
        img_dir = join(data, 'img').replace('\\', '/')
        if os.path.exists(stack_dir) and os.path.exists(img_dir):
            if 3 * len(os.listdir(img_dir)) == len(os.listdir(stack_dir)):
                continue

        stack_event(stack_name, index, data, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)


if __name__ == '__main__':
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

    train_root = os.path.join(target_dir, "train")
    test_root = os.path.join(target_dir, "test")

    if trans_folder == 1:
        if source_dir is None:
            print("Missing original dataset path")
            sys.exit()
        trans_pair(source_dir, target_dir)
        folder_trans(source_dir, target_dir)
        text_generate(target_dir)

    stack_dataset(train_root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)
    stack_dataset(test_root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)
