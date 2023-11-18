#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_time_memory.sh > run_time_memory.txt 2>&1 &
for seed in 1
do
    for attack_rate in 0.05 0.1 0.15 0.2
    #for attack_rate in 0.05
    do
        for model in deepwalk
        do 
            #for attack_method in viking dice minmax pgd clga aalp metattack
            for attack_method in aalp  nodeembeddingattack
            do
                for attack_goal in integrity availability
                do
                    {
                         for dataset in CiteSeer
                        do
                            mprof run -o ./time_memory_log/"--dataset $dataset --lp_model $model --attack_method $attack_method --exp_type poisoning --device cuda:0 --seed $seed --attack_rate $attack_rate --attack_goal $attack_goal.txt" main_time_memory.py --dataset $dataset --lp_model $model --attack_method $attack_method --exp_type poisoning --device cuda:0 --seed $seed --attack_rate $attack_rate --attack_goal $attack_goal
                        done
                    }
                done
            done
        done
    done
done