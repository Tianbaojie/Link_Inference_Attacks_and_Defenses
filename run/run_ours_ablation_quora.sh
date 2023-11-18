#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_ours_ablation_quora.sh > run_1114_quora.txt 2>&1 &
for dataset in quora
do 
    for ablation in ua uqta uqtqa
    do
        python main_attack_ablation.py --dataset $dataset --lp_model metatransformer --device cuda:0 --seed 42 --ablation_type $ablation
    done
done