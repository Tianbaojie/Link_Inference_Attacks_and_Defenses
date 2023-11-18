#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_clean_metamodel.sh > run_clean_5_6.txt 2>&1 &
for seed in 1
do 
    for dataset in Cora CiteSeer PubMed
    do
        for model in metamodel
        do
           {
              echo "python main.py --dataset $dataset --lp_model $model --exp_type clean --device cuda:0 --seed $seed" >> task_queue_clean_metamodel.txt
           }
        done
    done
done
# Execute tasks with limited concurrency
CONCURRENCY=4
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
done < task_queue_clean_metamodel.txt

# Wait for remaining tasks to complete
wait

cp task_queue_clean_metamodel.txt  task_queue_clean_metamodel_backup.txt
# Delete task queue file
rm task_queue_clean_metamodel.txt