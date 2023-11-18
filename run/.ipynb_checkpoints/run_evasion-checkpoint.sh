#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_evasion.sh > run_evasion_4_25.txt 2>&1 &
for dataset in Cora CiteSeer PubMed
do 
    for model in gcn gat
    do
        for attack_method in random dice metattack minmax pgd metattack aalp clga
        do
            for attack_rate in 0.05 0.1 0.15 0.2
            do
                for seed in 1 2 3 4 5 6 7 8 9 10
                do
                    python main.py --dataset $dataset --lp_model $model --attack_method $attack_method --exp_type evasion --device cuda:0 --seed $seed --attack_rate $attack_rate
                done
            done
        done
    done
done