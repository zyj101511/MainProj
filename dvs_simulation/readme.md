使用v2e转换FE108数据集的scripts_py调用流程:
 - `cvt_FE108_to_mp4.py` 将原数据集中的images转成mp4视频, 数据集结构不变
 - `move_FE108_txt.py` 将txt文件(ground truth)移动到新的目录'
 - `cvt_FE108_mp4_to_aedat2` 将mp4视频转换成aedat2格式
 -  使用DV GUI程序将所有aedat2文件转换成aedat4格式, 便于使用dv-processing进行后续处理
 - 