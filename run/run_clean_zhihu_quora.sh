#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_clean_zhihu_quora.sh > run_1108.txt 2>&1 &
for dataset in zhihu quora
do 
    for model in deepwalk gcn gat vgnae gcnii
    do
        for attack_method in noattack
        do
            for attack_rate in 0.05
            do
                for seed in 1
                do
                    python main.py --dataset $dataset --lp_model $model --attack_method $attack_method --exp_type clean --device cuda:1 --seed $seed --attack_rate $attack_rate
                done
            done
        done
    done
done