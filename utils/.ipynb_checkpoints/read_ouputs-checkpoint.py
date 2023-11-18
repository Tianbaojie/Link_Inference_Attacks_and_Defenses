import argparse
import os
import pickle
from os import path as osp

import numpy as np
import pandas as pd
import torch


def read_outputs():
    def change_tensor(best_val_result):
        for key in best_val_result.keys():
            if 'torch' in str(type(best_val_result[key])):
                best_val_result[key]=best_val_result[key].cpu()
        return np.array([i for i in best_val_result.values()])
    

    #获取文件夹outputs下的所有文件的路径
    current_path = os.path.abspath('.')
    #parent_path = os.path.dirname(current_path)
    outputs_path = osp.join(current_path, 'outputs')
    files = os.listdir(outputs_path)
    files_path = [osp.join(outputs_path, file) for file in files]
    
    #筛选出含有args的文件
    files_path = [file for file in files_path if 'args' in file]

    args_list = []
    for file in files_path:
        with open(file, 'rb') as f:
            args = pickle.load(f)
            args_list.append(args)
    
    best_test_result_file=[file[:-4]+'best_test_result' for file in files_path]
    best_test_result_list=[]
    for file in best_test_result_file:
        with open(file, 'rb') as f:
            best_test_result = pickle.load(f)
            best_test_result_list.append(change_tensor(best_test_result))
    
    best_val_result_file=[file[:-4]+'best_val_result' for file in files_path]
    best_val_result_list=[]
    for file in best_val_result_file:
        with open(file, 'rb') as f:
            best_val_result = pickle.load(f)
            best_val_result_list.append(change_tensor(best_val_result))
    
    #将args_list转换为dataframe
    data=pd.DataFrame([i.__dict__ for i in args_list])
    data['best_test_result']=best_test_result_list
    data['best_val_result']=best_val_result_list
    return data
    


#主函数

if __name__ == '__main__':
    data=read_outputs()
    data.to_csv('output.csv')
    data['val_auc']=data.best_val_result.apply(lambda x:round(x[0]*100,1))
    data['test_auc']=data.best_test_result.apply(lambda x:round(x[0]*100,2))
    
    data['val_acc']=data.best_val_result.apply(lambda x:round(x[1]*100,2))
    data['test_acc']=data.best_test_result.apply(lambda x:round(x[1]*100,2))
    
    data['val_recall']=data.best_val_result.apply(lambda x:round(x[2]*100,2))
    data['test_recall']=data.best_test_result.apply(lambda x:round(x[2]*100,2))
    
    data['val_precision']=data.best_val_result.apply(lambda x:round(x[3]*100,2))
    data['test_precision']=data.best_test_result.apply(lambda x:round(x[3]*100,2))
    
    data['val_f1']=data.best_val_result.apply(lambda x:round(x[4]*100,2))
    data['test_f1']=data.best_test_result.apply(lambda x:round(x[4]*100,2))
    
    clean=data.groupby(['dataset','lp_model','exp_type','attack_method','attack_goal','attack_rate'])[['val_auc','test_auc','val_acc','test_acc','val_recall','test_recall','val_precision','test_precision','val_f1','test_f1']].mean()
    clean=clean.apply(lambda x:round(x,1))
    clean.to_csv('output_result.csv')