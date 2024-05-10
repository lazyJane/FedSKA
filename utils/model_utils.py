import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import random
import numpy as np
from FLAlgorithms.trainmodel.models_jiu import Net, SimpleLinear
from FLAlgorithms.trainmodel.models import *
from torch.utils.data import DataLoader
from FLAlgorithms.trainmodel.RNN_traffic import *
from utils.model_config import *
from utils.constants import *
import scipy.sparse as sp
'''
METRICS = [
    'train_acc',
    'train_loss',
    'glob_acc', #test 
    'per_acc', 
    'glob_loss', 
    'per_loss'
    ]
'''
    
METRICS = [
    'train_acc',
    'train_loss',
    'glob_acc', #test
    'glob_acc_bottom_1', 
    'glob_acc_bottom_2', 
    'glob_acc_bottom_3', 
    'glob_acc_bottom_4', 
    'glob_acc_bottom_5', 
    'glob_acc_top_1', 
    'glob_acc_top_2', 
    'glob_acc_top_3', 
    'glob_acc_top_4', 
    'glob_acc_top_5', 
    'per_acc', 
    'glob_loss', 
    'per_loss', 
    'user_train_time',
    'server_agg_time',
    'total_train_time',
    'unseen_train_acc',
    'unseen_train_loss',
    'unseen_glob_acc',
    'unseen_glob_loss',
    'unseen_glob_acc_bottom',
    'adj',
    'train_mae',
    'train_mape',
    'train_mse',
    'glob_mae',
    'glob_mse',
    'glob_mape']
#METRICS = ['glob_acc', 'per_acc', 'glob_loss', 'per_loss','user_train_time', 'server_agg_time']

'''
METRICS = [
    'train_acc',
    'train_loss',
    'glob_acc', #test
    'glob_acc_bottom', 
    'per_acc', 
    'glob_loss', 
    'per_loss'
    ]
   
'''

def load_pretrain(model):
    #加载model，model是自己定义好的模型
    pretrain_model = get_mobilenet(n_classes = 10)
    
    #读取参数 
    pretrained_dict =pretrain_model.state_dict() 
    #print(pretrained_dict['classifier.1.weight'].shape)
    model_dict = model.state_dict() 
    #print(model_dict['classifier.1.weight'].shape)
    
    #将pretrained_dict里不属于model_dict的键剔除掉 
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
    
    # 更新现有的model_dict 
    model_dict.update(pretrained_dict) 
    
    # 加载我们真正需要的state_dict 
    model.load_state_dict(model_dict)  

    return model

def create_model_new(args):
    if 'metr-la' in args.dataset.lower():
        #print(args.algorithm)
        #if 'FedSKA' in args.algorithm:
            #model = GRUSeq2SeqWithGraphNet(input_size=2, output_size=2, hidden_size=args.user_feature_hidden_size, gru_num_layers=args.gru_num_layers,dropout=0).to(args.device), 'gruWithGraphNet'    
        #else:
        model = GRUSeq2Seq(input_size=2, output_size=2, hidden_size=args.user_feature_hidden_size, gru_num_layers=args.gru_num_layers,dropout=0).to(args.device), 'gru'
    if 'vehicle_sensor' in args.dataset.lower():
        model = DNN(100,20,2).to(args.device), 'dnn'
    elif 'femnist' in args.dataset.lower():
        print("femnist")
        model = FemnistCNN(num_classes=62).to(args.device), 'cnn'
    elif 'emnist' in args.dataset.lower():
        print('lll')
        model = FemnistCNN(num_classes=47).to(args.device), 'cnn'
    elif 'mnist' in args.dataset.lower():
        #model= Net('mnist', 'cnn').to(args.device), 'cnn'
        model= Mclr().to(args.device), 'mclr'
    elif 'cifar100' in args.dataset.lower():
        model = get_mobilenet(n_classes=100).to(args.device), 'cnn'
        
    elif 'cifar10' in args.dataset.lower():
        #model = MobilenetWithGraph(n_classes=10).to(args.device), 'cnn'
        if 'FedSKA' or 'Graph' in args.algorithm:
            model = MobilenetWithGraph(n_classes=10)
            model = load_pretrain(model)
            model = model.to(args.device), 'cnn'
        #model = CIFAR10CNN(num_classes=10).to(args.device),'cnn'
        else:
            model = get_mobilenet(n_classes=10).to(args.device), 'cnn'
    elif 'shakespeare' in args.dataset.lower():
        model =\
            NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"], #len=100
                embed_size=SHAKESPEARE_CONFIG["embed_size"], #8
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"], #256
                output_size=SHAKESPEARE_CONFIG["output_size"], #len=100
                n_layers=SHAKESPEARE_CONFIG["n_layers"] #2
            ).to(args.device), 'lstm'
    return model 

def get_log_path(args, algorithm, seed, gen_batch_size=32):
    #EMnist-alpha1.0-ratio0.5_FedKLtopk_0.01_10u_32b_20_0
    alg=args.dataset + "_" + algorithm
    alg+="_" + str(args.learning_rate) + "_" + str(args.num_users)
    if args.E != 0:
        alg+="u" + "_" + str(args.batch_size) + "b" + "_" + str(args.E)
    else:
        alg+="u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    alg=alg + "_" + str(seed)
    if 'FedGen' in algorithm: # to accompany experiments for author rebuttal
        alg += "_embed" + str(args.embedding)
        if int(gen_batch_size) != int(args.batch_size):
            alg += "_gb" + str(gen_batch_size)
    if 'batch_adj' in algorithm  or 'gcnadj' in algorithm or 'Graph' in algorithm :
        # bootstrap or not
        if args.tau != 0:
            alg += "_tau" + str(args.tau)
        if args.k != 0:
            alg += "_k" + str(args.k)
        if args.base != 'none':
            alg += "_base" + args.base
        if args.gen_mode != None:
            alg += "_gen_mode" + str(args.gen_mode)
        if args.graph_metric_type == 'gat_attention':
            alg += "_graph_metric_type" + str(args.graph_metric_type)
        if args.ratio > 0:
            alg += "_ratio" + str(args.ratio)
        #if 'Graph' in algorithm:
            #if args.gen_mode == 3:
                #alg += "_gen_mode" + str(args.gen_mode)
        # subgraph size k
        # 
        #  
        
    return alg


def sd_matrixing(state_dic, aggregate = False):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        if aggregate:
            if param_vector is None:
                param_vector = param.clone().detach().flatten()
            else:
                if len(list(param.size())) == 0:
                    param_vector = torch.cat((param_vector, param.clone().detach().view(1).type(torch.float32)), 0)
                else:
                    param_vector = torch.cat((param_vector, param.clone().detach().flatten()), 0)
        else:
            if 'classifier' in key or 'fc1'in key  or 'fc' in key :
                keys.append(key)
                if param_vector is None:
                    param_vector = param.clone().detach().flatten()
                else:
                    if len(list(param.size())) == 0:
                        param_vector = torch.cat((param_vector, param.clone().detach().view(1).type(torch.float32)), 0)
                    else:
                        param_vector = torch.cat((param_vector, param.clone().detach().flatten()), 0)
    return param_vector

'''
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
'''

def normalize_adj(adj, mode = "sym", sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())
    
def get_learning_rate(optimizer):
    for param in optimizer.param_groups:
        return param["lr"]
    