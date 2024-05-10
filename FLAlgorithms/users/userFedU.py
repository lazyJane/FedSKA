import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import copy
import numpy as np
from utils.model_utils import *
# Implementation for FedAvg clients

class UserFedU(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False)

        self.K = args.K
        self.L_K = args.L_K
        self.n_steps = 0

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        
        if self.E != 0: 
            self.n_steps = self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)

    def aggregate_parameters(self, user_list, global_in, num_clients, A):
        avg_weight_different = copy.deepcopy(list(self.model.parameters()))
        #akl = alk_connection
        for param in avg_weight_different:
            param.data = torch.zeros_like(param.data)
        
        A = normalize_adj(A) 
        # Calculate the diffence of model between all users or tasks
        for i in range(len(user_list)):
            if(self.id != user_list[i].id):
                #if(self.K > 0 and self.K <= 2):
                    #akl[int(self.id)][int(user_list[i].id)] = self.get_alk(user_list, dataset, i)
                # K == 3 : akl will be generate randomly for MNIST
                for avg, current_task, other_tasks in zip(avg_weight_different, self.model.parameters(),user_list[i].model.parameters()):
                    avg.data += A[int(self.id)][int(user_list[i].id)] * (current_task.data.clone() - other_tasks.data.clone())

        for avg, current_task in zip(avg_weight_different, self.model.parameters()):
            #print(current_task.data)
            #print(avg)
            current_task.data = current_task.data - self.learning_rate * self.L_K * avg



