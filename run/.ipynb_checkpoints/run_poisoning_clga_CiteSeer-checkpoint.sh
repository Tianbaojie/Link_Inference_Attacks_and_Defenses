#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_poisoning_clga_CiteSeer.sh > run_poisoning_clga_CiteSeer_5_6.txt 2>&1 &
export OPENBLAS_NUM_THREADS=1
for model in deepwalk node2vec gcn gat gca vgnae
do
    for seed in 1 2 3 4 5 6 7 8 9 10
    do
        {
            for dataset in  CiteSeer
            do 
                {
                     for attack_rate in 0.05 0.1 0.15 0.2
                    do
                        {
                            for attack_method in clga
                            do
                                {
                                    for attack_goal in integrity availability
                                    do
                                        {
                                             echo "python3 main.py --dataset $dataset --lp_model $model --attack_method $attack_method --exp_type poisoning --device cuda:0 --seed $seed --attack_rate $attack_rate --attack_goal $attack_goal" >> task_queue_clga_CiteSeer.txt
                                        }
                                    done
                                }
                            done
                        }
                    done
                }
            done
        }
    done
done

# Execute tasks with limited concurrency
CONCURRENCY=8
TASKS_RUNNING=0

while read task || [[ -n "$task" ]]
do
    $task &
    TASKS_RUNNING=$(expr $TASKS_RUNNING + 1)
    if [ $TASKS_RUNNING -ge $CONCURRENCY ]
    then
        wait -n
        TASKS_RUNNING=$(expr $TASKS_RUNNING - 1)
    fi
done < task_queue_clga_CiteSeer.txt

# Wait for remaining tasks to complete
wait
cp task_queue_clga_CiteSeer.txt task_queue_clga_CiteSeer_backup.txt
# Delete task queue file
rm task_queue_clga_CiteSeer.txt