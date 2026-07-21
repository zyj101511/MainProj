CUDA_VISIBLE_DEVICES=0,1 python3 -m tracking.train --script_name train_script_plain_CA --config_name fe108_mastrack --save_dir ./output \
--mode multiple --nproc_per_node 2