import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.optimizers.fedoptimizer import FedProxOptimizer,ProxSGD
from FLAlgorithms.users.userbase import User
import copy
# Implementation for pFeMe clients

class UserpFedMe(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False)

        #self.loss = nn.NLLLoss()
        self.personal_learning_rate = args.personal_learning_rate
        if 'mnist' == self.dataset[0]:
            self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda,momentum=0.9)
        else:
            self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda,  mu = 0.001, momentum=0.9)
        
    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        if self.E ==0:
            local_epochs = self.local_epochs
        else:
            if self.train_samples < self.batch_size:
                local_epochs = 1
            else:
                local_epochs = int(self.E * (self.train_samples / self.batch_size ))
        for epoch in range(1, local_epochs + 1):  # local update
            
            self.model.train()
            result = self.get_next_train_batch()
            X, y = result['X'], result['y']
            X, y = X.to(self.device), y.to(self.device)

            # K = 30 # K is number of personalized steps
            for i in range(self.K):
                self.optimizer.zero_grad()
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:   
                    output = self.model(X)
                    loss = self.ce_loss(output, y)
                else: 
                    output = self.model(X)['output']
                    loss = self.loss(output, y)
                loss.backward()
                #把localmodel传进去
                self.personalized_model_bar, _ = self.optimizer.step(self.local_model) # group['params']:\theta

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.personalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)
                #w=w-\ita\lamda(w-\theta)

        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(self.local_model) # 用local_model更新self.model

        return LOSS