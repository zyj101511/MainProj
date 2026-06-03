import cv2
import os
from glob import glob

img_dir = "/path/to/images"
out_path = "output.mp4"

# 按文件名排序
img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))

# 读第一张确定尺寸
first = cv2.imread(img_paths[0])
h, w = first.shape[:2]

# 视频编码
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, 30, (w, h))

for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        continue
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    writer.write(img)

writer.release()
print("saved to", out_path)
