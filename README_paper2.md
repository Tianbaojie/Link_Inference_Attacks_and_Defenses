#  Explainable Graph Transformer: An Explainable Approach for Link Inference Attacks based on Meta Paths
 Official implementation of Explainable Graph Transformer: An Explainable Approach for Link Inference Attacks based on Meta Paths


Built based on [GCA](https://github.com/CRIPAC-DIG/GCA), [DeepRobust](https://deeprobust.readthedocs.io/en/latest/#), [Viking](https://github.com/virresh/viking), [CLGA](https://github.com/RinneSz/CLGA)

## Requirements
Tested on pytorch 1.7.1 and torch_geometric 1.6.3.

## Usage
1.To produce availability attack with STAA
```
python main.py --dataset Cora --attack_method aalp --exp_type poisoning --device cuda:0 --attack_rate 0.05 --seed 2 --lp_model deepwalk  --attack_goal='availability'
```

2.To produce Integrity attack with STAA
```
python main.py --dataset CiteSeer --attack_method aalp --exp_type poisoning --device cuda:0 --attack_rate 0.1 --seed 2 --lp_model deepwalk  --attack_goal='integrity'
```

3.To run clean experiments
```
python main.py --dataset Cora --lp_model metamodel --exp_type clean --device cuda:0 --seed 1
```
python main.py --dataset Quora --lp_model gcn --exp_type clean --device cuda:0 --seed 1

4.To produce all experiments, pealse config scripts in ./run/ 
```
bash ./run/run_poisoning.sh 
```
