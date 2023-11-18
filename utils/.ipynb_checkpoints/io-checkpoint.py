import os
import os.path as osp
import pickle

import torch

from utils.utils import generate_dataset


def load_dataset(dataset):
    path = osp.expanduser('dataset')
    path = osp.join(path, dataset)
    #判断是否向存在
    if not osp.exists(osp.join(path, 'data.pt')):
        generate_dataset(dataset)
    data = torch.load(osp.join(path, 'data.pt'))
    return data

#使用pickle将模型保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed组成
def save_model(model, args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal,  str(args.attack_rate), str(args.seed)]))
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

#使用pickle将clean模型从args.outputs文件夹下读取出来,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed组成
def load_clean_model(args):
    filename = osp.join(args.outputs, '_'.join([args.dataset, 'clean', args.lp_model, 'noattack','integrity',  '0.05', str(args.seed)]))
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model
#使用pickle将模型的best_val_result结果保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed和best_val_result组成
def save_best_val_result(best_val_result, args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal, str(args.attack_rate), str(args.seed), 'best_val_result']))
    with open(filename, 'wb') as f:
        pickle.dump(best_val_result, f)

#使用pickle将模型的best_test_result结果保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed和best_test_result组成
def save_best_test_result(best_test_result, args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal,str(args.attack_rate), str(args.seed), 'best_test_result']))
    with open(filename, 'wb') as f:
        pickle.dump(best_test_result, f)

#使用pickle将模型的best_scores结果保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed和best_scores组成
def save_best_scores(best_scores, args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal, str(args.attack_rate), str(args.seed), 'best_scores']))
    with open(filename, 'wb') as f:
        pickle.dump(best_scores, f)

#使用pickle将args保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed和args组成
def save_args(args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method, args.attack_goal,str(args.attack_rate), str(args.seed), 'args']))
    with open(filename, 'wb') as f:
        pickle.dump(args, f)

#保存model, best_val_result, best_test_result, best_scores, args, data
def save_results(model, best_val_result, best_test_result, best_scores, args, data):
    if args.lp_model=='gcn' or args.lp_model=='gat':
        save_model(model, args)
    save_best_val_result(best_val_result, args)
    save_best_test_result(best_test_result, args)
    #save_best_scores(best_scores, args)
    save_args(args)

#读取modifiedAdj.pkl
def load_modifiedAdj(args):
    if args.dataset=='zhihu' or args.dataset=='quora':
        filename=f'{args.outputs}/adj/{args.dataset}_{args.attack_method}_{args.attack_goal}_{args.attack_rate}_modifiedData.pkl'
    else:
        filename=f'{args.outputs}/adj/{args.dataset}_{args.attack_method}_{args.attack_goal}_{args.attack_rate}_modifiedAdj.pkl'
    with open(filename, 'rb') as f:
        modifiedAdj = pickle.load(f)
    return modifiedAdj