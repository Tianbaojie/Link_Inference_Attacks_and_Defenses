#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_poisoning_aalp_PubMed.sh > run_poisoning_5_6.txt 2>&1 &
export OPENBLAS_NUM_THREADS=1
for seed in 1 2 3 4 5 6 7 8 9 10
do
    for attack_rate in 0.05 0.1 0.15 0.2
    do
        for model in deepwalk node2vec gcn gat gca vgnae
        do 
            for attack_method in aalp
            do
                for attack_goal in integrity availability
                do
                    {
                         for dataset in PubMed
                        do
                            {
                                 echo "python3 main.py --dataset $dataset --lp_model $model --attack_method $attack_method --exp_type poisoning --device cuda:0 --seed $seed --attack_rate $attack_rate --attack_goal $attack_goal" >> task_run_poisoning_aalp_PubMed.txt
                            }
                        done
                    }
                done
            done
        done
    done
done


# Execute tasks with limited concurrency
CONCURRENCY=1
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
done < task_run_poisoning_aalp_PubMed.txt

# Wait for remaining tasks to complete
wait
cp task_run_poisoning_aalp_PubMed.txt task_run_poisoning_aalp_PubMed_backup.txt
# Delete task queue file
rm task_run_poisoning_aalp_PubMed.txt