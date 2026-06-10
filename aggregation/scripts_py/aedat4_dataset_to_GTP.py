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

np.set_printoptions(suppress=True)  # 设置 numpy 打印选项，避免科学计数法显示

pair = {}  # 定义一个字典，用于存储序列名称与起始帧的对应关系


def get_start_frame(seq_name):
    """获取给定序列的起始帧号"""

    return pair[seq_name]  # 返回对应的起始帧号


def stack_event(stack_name, index, root, pair_root, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    """
    处理指定序列的事件数据，并将其存储为图像。
    参数:
    - index: 当前处理的序列索引
    - root: 序列的根目录路径
    """
    match_file = pair_root  # 定义配对文件的名称
    with open(match_file, 'r') as f:  # 打开配对文件
        for line in f.readlines():  # 逐行读取文件
            file, start_frame = line.split()  # 获取文件名和起始帧
            pair[file] = int(start_frame) + 1  # 存储起始帧到 pair 字典中，值加1以便从下一帧开始处理

    root = root  # 当前序列的根目录路径
    seq_name = root.split('/')[-1]  # 获取序列名称                                  airplane_mul222

    img_path = os.path.join(root, 'img').replace('\\',
                                                 '/')  # 构建图像路径                            ./test/airplane_mul222/img
    frame_num = len(os.listdir(img_path))  # 获取图像帧的数量
    event_data = os.path.join(root, 'events.aedat4').replace('\\', '/')  # 构建事件数据文件的路径
    stack_path = os.path.join(root, stack_name).replace('\\',
                                                            '/')  # 定义保存事件栈的路径         ./test/airplane_mul222/inter3_stack

    start_frame = get_start_frame(seq_name)  # 获取当前序列的起始帧

    if not os.path.exists(stack_path):  # 如果保存路径不存在
        os.mkdir(stack_path)  # 创建该目录

    with AedatFile(event_data) as f:  # 打开 AEDAT4 事件数据文件
        pic_shape = f['events'].size  # 获取事件数据的尺寸
        events = np.hstack([packet for packet in f['events'].numpy()])  # 将事件数据平铺成二维数组
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']  # 时间戳、x、y、极性
        event = np.vstack((timestamps, x, y, polarities))  # 将这些属性堆叠为一个二维数组
        event = np.swapaxes(event, 0, 1)  # 交换数组的轴，便于后续处理

        # event = [
        #     [timestamp1, x1, y1, polarity1],
        #     [timestamp2, x2, y2, polarity2],
        #     ...
        #     [timestampN, xN, yN, polarityN]
        # ]

        time_series = []  # 初始化一个列表，用于存储帧的时间戳
        count = 0  # 帧计数器
        for frame in f["frames"]:  # 遍历每个帧
            count += 1  # 增加计数
            if count >= start_frame and count <= start_frame + frame_num:  # 检查当前帧是否在起始帧和总帧数之间
                time_series.append(frame.timestamp_start_of_frame)  # 记录帧的起始时间戳
            else:
                continue  # 否则继续下一帧

        # 筛选事件时间戳在起始帧到结束帧范围内的事件
        event = event[event[:, 0] >= time_series[0]]
        event = event[event[:, 0] < time_series[-1]]

        # 处理并保存事件数据
        deal_event(index, event, time_series, pic_shape, stack_path, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)


def process_event(pos_img, neg_img, null_img, event, pic_shape, stack_amount_1c2c):
    """
    将单个事件信息处理到图像上。
    参数:
    - pos_img: 事件图像
    - event: 当前事件
    - pic_shape: 图片尺寸
    """
    x, y, p = int(event[1]), int(event[2]), int(event[3])  # 获取事件的坐标 (x, y) 和极性 p
    if 0 < x < pic_shape[1] and 0 < y < pic_shape[0]:  # 检查坐标是否在图像范围内
        if p == 1:  # 如果极性为 1
            pos_img[y][x] = min(255, pos_img[y][x] + stack_amount_1c2c)  # 将对应像素设置为黑色
        else:
            neg_img[y][x] = min(255, neg_img[y][x] + stack_amount_1c2c)  # 否则设置为白色

def save_2C_img(pos_img, neg_img, null_img, root):
    # 创建一个两通道的图像

    two_channel_img = np.zeros((pos_img.shape[0], pos_img.shape[1], 3), dtype=np.uint8)
    two_channel_img[:, :, 0] = pos_img  # 将第一个通道设置为 pos_img
    two_channel_img[:, :, 1] = neg_img  # 将第二个通道设置为 neg_img
    two_channel_img[:, :, 2] = null_img

    # df = pd.DataFrame(two_channel_img[:, :, 0])
    # df.to_csv('/data/dataset/FE108_3C/png_test.csv', index=False, header=False)

    cv2.imwrite(root, two_channel_img)


def save_1C_img(img, root):
    two_channel_img = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    two_channel_img[:, :, 0] = img  # 将通道设置为 img
    cv2.imwrite(root, two_channel_img)


def hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_pic, stack_amount_3c, decay_rate_3c):
    """
    生成新的隐藏图片。

    参数：
    - pos_img: 当前的正图像，单通道，2维ndarray
    - neg_img: 当前的负图像，单通道，2维ndarray
    - hidden_pic: 当前的隐藏图像，单通道，2维ndarray
    - last_pos_pic: 上一次的正图像，单通道，2维ndarray
    - last_neg_pic: 上一次的负图像，单通道，2维ndarray
    - last_hidden_pic: 上一次的隐藏图像，单通道，2维ndarray
    - alpha: 权重系数，浮点数
    - rate: 增量值，浮点数

    返回：
    - 新的隐藏图片，单通道，2维ndarray
    """
    # 初始化新的hidden_pic数组
    new_hidden_pic = hidden_pic * decay_rate_3c

    # 条件1：last_pos_pic为0且pos_img不为0
    pos_condition = (last_pos_pic == 0) & (pos_img != 0)

    # 条件2：last_neg_pic为0且neg_img不为0
    neg_condition = (last_neg_pic == 0) & (neg_img != 0)

    # 满足条件的像素位置增加rate
    new_hidden_pic[pos_condition] += stack_amount_3c
    new_hidden_pic[neg_condition] += stack_amount_3c

    new_hidden_pic = np.clip(new_hidden_pic, 0, 255)

    return new_hidden_pic

def deal_event(index, events, frame_timestamp, pic_shape, save_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    """
    处理事件数据，将它们分段保存为图像。
    参数:
    - index: 当前序列索引
    - events: 事件数据
    - frame_timestamp: 帧时间戳
    - pic_shape: 图片尺寸
    - save_name: 保存路径
    """

    flag = False
    last_pos_pic = np.full(pic_shape, 0, dtype=np.uint8)
    last_neg_pic = np.full(pic_shape, 0, dtype=np.uint8)
    hidden_state = np.full(pic_shape, 0, dtype=np.uint8)

    i = 1  # 帧索引
    pos_img = np.full(pic_shape, 0, dtype=np.uint8)  # 初始化灰色背景图像
    neg_img = np.full(pic_shape, 0, dtype=np.uint8)  # 初始化灰色背景图像
    sub_index = 1  # 子帧索引
    T_num = 2
    sub_frame = np.linspace(frame_timestamp[0], frame_timestamp[1], T_num)  # 将帧时间戳等分为 4 个子帧

    pos_eve = 0
    neg_eve = 0
    other_eve = 0
    error_eve = 0
    # for event in tqdm(events, desc="{} Writing {} events ".format(index, save_name.split('/')[-2])):
    #     if(event[3] == 1):
    #         pos_eve = pos_eve + 1
    #     elif(event[3] == 0):
    #         neg_eve = neg_eve + 1
    #     else:
    #         other_eve = other_eve + 1
    #
    #     if(event[1] == 86 and event[2] == 236):
    #         error_eve = error_eve + 1

    # print("num of pos,neg,other events", pos_eve, neg_eve, other_eve)
    # 遍历每个事件
    for event in tqdm(events, desc="{} Writing {} events ".format(index, save_name.split('/')[-2])):
        # if(i == 1114):
        #     print("1114")
        # np.savetxt('/data/dataset/FE108_3C/event.txt', events, fmt='%d', delimiter=' ', header="t    x   y   p", comments='')
        if event[0] >= frame_timestamp[i]:  # 如果事件时间大于当前帧时间
            img_save_root = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
            # img_save_root = '/data/dataset/FE108_3C_2/train/airplane/save_for_test'+'/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
            # if(img_save_root == '/data/dataset/FE108_3C/train/airplane/inter3_stack_stack1/1114_3.png'):
            #     print("pause")
            if flag == False:
                flag = True
            else:
                hidden_state = hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c, decay_rate_3c)
            last_pos_pic = pos_img
            last_neg_pic = neg_img

            save_2C_img(pos_img, neg_img, hidden_state, img_save_root)
            # save_1C_img(pos_img, img_save_root)

            i = i + 1  # 更新帧索引

            sub_frame = np.linspace(frame_timestamp[i - 1], frame_timestamp[i], T_num)  # 重新计算子帧范围
            pos_img = np.full(pic_shape, 0, dtype=np.uint8)  # 初始化灰色背景图像
            neg_img = np.full(pic_shape, 0, dtype=np.uint8)  # 初始化灰色背景图像  # 重置图像

            sub_index = 1  # 重置子帧索引
        elif event[0] < frame_timestamp[i]:
            # count += 1
            if event[0] >= sub_frame[sub_index]:
                # event_count_list.append(count)
                # count = 0
                # cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png', pos_img)  # 保存图像

                img_save_root = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
                # img_save_root = '/data/dataset/FE108_3C_2/train/airplane/save_for_test'+'/' + str(i).zfill(4)+ '_' + str(sub_index) + '.png'
                # if (img_save_root == '/data/dataset/FE108_3C/train/airplane/inter3_stack_stack1/1114_3.png'):
                #     print("pause")
                if flag == False:
                    flag = True
                else:
                    hidden_state = hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c, decay_rate_3c)
                last_pos_pic = pos_img
                last_neg_pic = neg_img


                save_2C_img(pos_img, neg_img, hidden_state, img_save_root)
                # save_1C_img(pos_img, img_save_root)

                pos_img = np.full(pic_shape, 0, dtype=np.uint8)  # 初始化灰色背景图像
                neg_img = np.full(pic_shape, 0, dtype=np.uint8)  # 初始化灰色背景图像  # 重置图像

                sub_index = sub_index + 1  # 更新子帧索引
            process_event(pos_img, neg_img, hidden_state, event, pic_shape, stack_amount_1c2c)  # 处理事件

    # cv2.imwrite(save_name + '/' + str(i).zfill(4) + '_' + str(3) + '.png', pos_img)  # 保存最后一帧图像
    img_save_root = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
    # img_save_root = '/data/dataset/FE108_3C_2/train/airplane/save_for_test'+'/' + str(i).zfill(4)+ '_' + str(sub_index) + '.png'
    hidden_state = hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c,
                                            decay_rate_3c)
    save_2C_img(pos_img, neg_img, hidden_state, img_save_root)

def trans_pair(source_dir, target_dir):
    # 确保目标目录存在，如果不存在则创建
    os.makedirs(target_dir, exist_ok=True)

    # 源文件路径
    source_file = os.path.join(source_dir, 'pair.txt')

    # 目标文件路径
    target_file = os.path.join(target_dir, 'pair.txt')

    # 检查源文件是否存在
    if os.path.isfile(source_file):
        # 复制文件到目标目录
        shutil.copy(source_file, target_file)
        print(f"Copied {source_file} to {target_file}")
    else:
        print(f"Source file {source_file} does not exist.")

def folder_trans(source_dir, target_dir):  # 复制数据集除inter3_stack以外的文件
    # 遍历源目录中的train和test文件夹
    for split_dir in os.listdir(source_dir):
        if split_dir in ["train", "test"]:
            split_source_dir = os.path.join(source_dir, split_dir)
            split_target_dir = os.path.join(target_dir, split_dir)
            # 创建目标目录（如果不存在）
            if not os.path.exists(split_target_dir):
                os.makedirs(split_target_dir)

            # 遍历train或test文件夹下的子文件夹
            for sub_dir in os.listdir(split_source_dir):
                if sub_dir in ["test.txt", "train.txt"]:
                    continue
                if sub_dir != "inter3_stack":  # 排除inter3_stack文件夹
                    sub_source_dir = os.path.join(split_source_dir, sub_dir)
                    sub_target_dir = os.path.join(split_target_dir, sub_dir)

                    # 复制除inter3_stack文件夹外的所有内容到目标目录
                    shutil.copytree(sub_source_dir, sub_target_dir, ignore=shutil.ignore_patterns('inter3_stack'))

                    print(f"复制 {sub_dir} 完成")
    print("数据集复制完成")

def text_generate(target_dir):  # 复制text文件
    # 指定train的text
    text_root = os.path.join(target_dir, "train")
    # 获取train文件夹下所有文件夹的名称
    folder_names = [name for name in os.listdir(text_root) if os.path.isdir(os.path.join(text_root, name))]
    # 生成一个txt文件并写入所有文件夹名
    output_file = os.path.join(text_root, "test.txt")
    with open(output_file, 'w') as file:
        for name in folder_names:
            file.write(name + '\n')
    print("文件夹名已写入到", output_file, "文件中。")
    # 指定test的text
    text_root = os.path.join(target_dir, "test")
    # 获取train文件夹下所有文件夹的名称
    folder_names = [name for name in os.listdir(text_root) if os.path.isdir(os.path.join(text_root, name))]
    # 生成一个txt文件并写入所有文件夹名
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

    parser.add_argument('--stack_amount_1c2c', type=float, default=40, help='Value of single stacking for the first two channels')
    parser.add_argument('--stack_amount_3c', type=float, default=64, help='Value of single stacking for the third channel (hidden_pic)')
    parser.add_argument('--decay_rate_3c', type=float, default=0.8, help='Decay of values between two frames for the third channel (hidden_pic)')

    args = parser.parse_args()
    return args


def stack_dataset(root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    # root:/data/dataset/FE108_rgb/train或test
    file_name_list = []
    text_root = os.path.join(root, "test.txt")
    pair_root = os.path.join(target_dir, "pair.txt")
    # 打开文件，读取验证集的文件名列表
    with open(text_root, 'r') as f:
        a = [i.strip() for i in f.readlines()]  # 去除每行的空白字符
        for line in a:
            file_name_list.append(line)  # 将每行文件名添加到列表中
    # 遍历文件名列表
    # for index, i in enumerate(sorted(file_name_list)[s_train:e_train]):  # 对108个文件夹进行循环
    for index, i in enumerate(sorted(file_name_list)[:]):  # 对108个文件夹进行循环
        data = os.path.join(root, i).replace('\\', '/')  # 值为第i个序列的主文件夹路径
        # data = "/data/users/wuhd/code/stack_event/truck_motion"
        if os.path.exists(join(data, stack_name).replace('\\', '/')):  # 如果目标目录已经存在
            if 3 * len(os.listdir(join(data, 'img').replace('\\', '/'))) == len(
                    os.listdir(join(data, stack_name).replace('\\', '/'))):
                continue  # 跳过已经处理的文件
        stack_event(stack_name, index, data, pair_root, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)  # 处理事件数据


# config设置,复制FE108数据集到FE108_rgb_stack，并完成stack：--trans_folder 1 --source_dir /data/dataset/FE108 --target_dir /data/dataset/FE108_rgb_stack --s_train 0 --e_train 76 --s_test 0 --e_test 32 --stack_amount_1c2c 40 --stack_amount_3c 64 --decay_rate_3c 0.8

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

# 复制数据集
if (trans_folder == 1):
    if source_dir is None:
        print("Missing original dataset path")
        sys.exit()
    trans_pair(source_dir, target_dir)
    folder_trans(source_dir, target_dir)
    text_generate(target_dir)
# 堆叠事件流
stack_dataset(train_root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)
stack_dataset(test_root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)