#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_poisoning_aalp.sh > run_poisoning_5_6.txt 2>&1 &
export OPENBLAS_NUM_THREADS=1
for seed in 1
do
    for attack_rate in 0.05 0.1 0.15 0.2
    do
        for model in metamodel
        do 
            for attack_method in aalp
            do
                for attack_goal in availability
                do
                    {
                         for dataset in Cora CiteSeer
                        do
                            {
                                 echo "python3 main.py --dataset $dataset --lp_model $model --attack_method $attack_method --exp_type poisoning --device cuda:0 --seed $seed --attack_rate $attack_rate --attack_goal $attack_goal" >> task_queue_run_poisoning_aalp.txt
                            }
                        done
                    }
                done
            done
        done
    done
done


# Execute tasks with limited concurrency
CONCURRENCY=2
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
done < task_queue_run_poisoning_aalp.txt

# Wait for remaining tasks to complete
wait
cp task_queue_run_poisoning_aalp.txt task_queue_run_poisoning_aalp_backup.txt
# Delete task queue file
rm task_queue_run_poisoning_aalp.txt