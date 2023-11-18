import argparse
import os
import pickle
from os import path as osp

import numpy as np
import pandas as pd
import torch
import re
import datetime
def read_outputs():
    #获取文件夹outputs下的所有文件的路径
    current_path = os.path.abspath('.')
    #parent_path = os.path.dirname(current_path)
    outputs_path = osp.join(current_path, 'time_memory_log')
    files = os.listdir(outputs_path)
    files_path = [osp.join(outputs_path, file) for file in files]
    
    #筛选出含有args的文件
    files_path = [file for file in files_path if '--dataset' in file]
    print(files_path[15])
    args_list,text_list,cuda_list = [],[],[]
    for file in files_path:
        with open(file, 'r') as f:
            text = f.read()
            text_list.append(text)
        pattern = r'--(\w+)\s+([\w.:]+)'
        # Find all matches in the command string
        matches = re.findall(pattern,file[:-4])
        # Create a dictionary of argument names and their values
        args = dict(matches)
        process_name='_'.join([args['dataset'], args['exp_type'], args['lp_model'], args['attack_method'],args['attack_goal'],str(args['attack_rate']), str(args['seed']), 'cuda'])
        file_name="GPU_stat_"+process_name+".txt"
        with open(osp.join(outputs_path, file_name), 'r') as f:
            text = f.read()
            cuda_list.append(text)
    for text,cuda in zip(text_list,cuda_list):
        lines=text.split('\n')
        args = dict()
        for line in lines:
            if line.startswith('FUNC'):
                values=line.split()[-5:]
                time2=datetime.datetime.fromtimestamp(eval(values[3]))
                time1=datetime.datetime.fromtimestamp(eval(values[1]))
                run_time=(time2 - time1).total_seconds()
                space=abs(eval(values[2])-eval(values[0]))
                args.update({'run_time':round(run_time,2),'space':round(space,2)})
            elif line.startswith('CMDLINE'):
                pattern = r'--(\w+)\s+([\w.:]+)'
                # Find all matches in the command string
                matches = re.findall(pattern, line)
                # Create a dictionary of argument names and their values
                args.update(dict(matches))
        lines=cuda.split('\n')
        values=[]
        for line in lines[1:-1]:
            value=line.split('\t')
            values.append(eval(value[-2]))
            
        args.update({'cuda':max(values)})
                
        
        args_list.append(args)           
    #将args_list转换为dataframe
    data=pd.DataFrame(args_list)
    return data
    
#主函数

if __name__ == '__main__':
    data=read_outputs()
    data.to_csv('output_time_memory.csv')