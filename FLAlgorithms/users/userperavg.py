import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD
from FLAlgorithms.users.userbase import User

# Implementation for Per-FedAvg clients

class UserPerAvg(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False)
        
        
        #if(model[1] == "Mclr_CrossEntropy"):
            #self.loss = nn.CrossEntropyLoss()
        #else:
            #self.loss = nn.NLLLoss()
        if 'mnist' == self.dataset[0]:
            self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate,  momentum=0.9)
            #print('momentum')

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        #self.model = copy.deepcopy()
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)

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

            temp_model = copy.deepcopy(list(self.model.parameters()))

            #step 1
            result = self.get_next_train_batch()
            X, y = result['X'], result['y']
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:
                output = self.model(X)
                loss = self.ce_loss(output, y)
            else: 
                output = self.model(X)['output']
                loss = self.loss(output, y)
            loss.backward() #计算新下载下来的模型在D上的梯度
            self.optimizer.step() #用这个梯度更新参数

            #step 2
            result = self.get_next_train_batch()
            X, y = result['X'], result['y']
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:
                output = self.model(X)
                loss = self.ce_loss(output, y)
            else: 
                output = self.model(X)['output']
                loss = self.loss(output, y)
            loss.backward() #计算新参数在D'上的梯度

            # restore the model parameters to the one before first update
            for old_p, new_p in zip(self.model.parameters(), temp_model):
                old_p.data = new_p.data.clone()
                
            self.optimizer.step(beta = self.beta) #计算最终的参数

            # clone model to user model 
            self.clone_model_paramenter(self.model.parameters(), self.local_model)

        return LOSS    

    def train_one_step(self):
        self.model.train()
        #step 1
        if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:
            result = self.get_next_train_batch()
        else:
            result = self.get_next_train_batch()
        X, y = result['X'], result['y']
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:
            output = self.model(X)
            loss = self.ce_loss(output, y)
        else: 
            output = self.model(X)['output']
            loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
            #step 2
        if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:  
            result = self.get_next_train_batch()
        else:
            result = self.get_next_train_batch()
        X, y = result['X'], result['y']
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
            output = self.model(X)
            loss = self.ce_loss(output, y)
        else: 
            output = self.model(X)['output']
            loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)

    def train_unseen(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        #self.model = copy.deepcopy()
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True, unseen = True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, unseen = True)