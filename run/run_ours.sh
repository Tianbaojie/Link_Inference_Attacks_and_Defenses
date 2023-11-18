#!/bin/bash
#在所有数据集上运行实验
#nohup bash ./run/run_rgcnn.sh > run_1108.txt 2>&1 &
python main_attack.py --dataset quora --lp_model metatransformer --device cuda:1 --seed 42
python main_attack.py --dataset zhihu --lp_model metatransformer --device cuda:1 --seed 42