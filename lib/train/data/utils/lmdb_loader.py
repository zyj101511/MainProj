import lmdb
import numpy as np
import cv2
import json
import io

LMDB_ENVS = dict()
LMDB_HANDLES = dict()
LMDB_FILELISTS = dict()


def get_lmdb_handle(name):
    """
    不要每次读一个 key 都重新 lmdb.open(...)
    减少重复打开数据库的开销
    让整个进程里对同一个 LMDB 路径复用同一个只读 handle
    """
    global LMDB_HANDLES, LMDB_FILELISTS
    item = LMDB_HANDLES.get(name, None)
    if item is None:
        env = lmdb.open(name, readonly=True, lock=False, readahead=False, meminit=False)
        LMDB_ENVS[name] = env
        item = env.begin(write=False)
        LMDB_HANDLES[name] = item

    return item


def decode_img(lmdb_fname, key_name):
    handle = get_lmdb_handle(lmdb_fname)
    binfile = handle.get(key_name.encode())
    if binfile is None:
        print(f"Illegal data detected. {lmdb_fname}, {key_name}")
    s = np.frombuffer(binfile, np.uint8)
    x = cv2.imdecode(s, cv2.IMREAD_COLOR)
    return x


def decode_str(lmdb_fname, key_name):
    handle = get_lmdb_handle(lmdb_fname)
    binfile = handle.get(key_name.encode())
    string = binfile.decode()
    return string

def decode_txt(lmdb_fname, key_name):
    string = decode_str(lmdb_fname, key_name)
    array = np.loadtxt(io.StringIO(string), delimiter=',', dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]  # 保证返回二维
    return array


def decode_json(lmdb_fname, key_name):
    return json.loads(decode_str(lmdb_fname, key_name))


if __name__ == "__main__":
    for i in range(100):
        lmdb_fname = "/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_nbinsGTP_lmdb"
        '''Decode image'''
        meta = decode_json(lmdb_fname, "train/meta.json")
        video_name = meta["videos"][0]['video_name']
        frame_name = meta["videos"][0]['frame_names'][i]
        num_frames = meta["videos"][0]['num_frames']
        key_name = f"train/{video_name}/{frame_name}"

        img = decode_img(lmdb_fname, key_name)
        gt = decode_txt(lmdb_fname, f"train/{video_name}/gt.txt")[i]
        print(gt)

        print(num_frames)
        cv2.rectangle(img, (int(gt[0]), int(gt[1])), (int(gt[0])+int(gt[2]), int(gt[1])+int(gt[3])),
                      color = (0, 0, 255), thickness=1)
        cv2.imshow("img", img)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
