使用v2e转换FE108数据集的scripts_py调用流程:
 - `FE108_imgs_to_mp4.sh` 将原数据集中的images转成mp4视频, 数据集结构不变
 - `move_FE108__gt_txt.sh` 将txt文件(ground truth)移动到新的目录'
 - `FE108_mp4_to_aedat4.sh` 将mp4视频转换成aedat4格式