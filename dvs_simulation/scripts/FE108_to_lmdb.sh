python3 /home/yanjiezhang/Downloads/Dissertation/MainProj/lib/train/data/img_to_lmdb.py \
--src_root /home/yanjiezhang/Downloads/Dissertation/dataset/FE108 \
--dst_lmdb /home/yanjiezhang/Downloads/Dissertation/dataset/FE108_nbinsGTP_lmdb \
--sub_dir_name imgs_nbinsGTP \
--gt_name groundtruth_rect.txt \
--map_size 64 * 1024 ** 3