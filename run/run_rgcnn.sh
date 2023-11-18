#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_rgcnn.sh > run_1108.txt 2>&1 &
for dataset in zhihu quora
do 
    for model in rgcn
    do
        python main_attack.py --dataset $dataset --lp_model $model  --device cuda:1 --seed 42
    done
done