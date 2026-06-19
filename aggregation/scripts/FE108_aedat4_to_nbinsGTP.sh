cd /home/yanjiezhang/Downloads/Dissertation/MainProj
python3 -m aggregation.scripts_py.aedat4_dataset_to_nbinsGTP \
        -d /media/yanjiezhang/ian/dataset/FE108_raw \
        -f /media/yanjiezhang/ian/dataset/FE108_raw/offset.txt \
        -c \
        -o /media/yanjiezhang/ian/dataset/FE108/ \
        --ch12_strength 40 \
        --ch3_strength 30 \
        --ch3_decay_rate 0.9 \
        --num_bins 4