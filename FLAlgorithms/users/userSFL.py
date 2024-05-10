from fcntl import F_SETLEASE
import torch
from FLAlgorithms.users.userbase import User
import copy
from utils.model_utils import *

class UserSFL(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False)
        self.global_param = self.model.state_dict()
        self.server_param = self.model.state_dict()
        
        self.mode = "Train"


    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        unseen = False
        if self.E != 0: 
            if 'shakespeare' in self.dataset: 
                self.fit_epochs_shakespeare(glob_iter, lr_decay)
                return 
            if 'metr' in self.dataset:
                self.fit_epochs_metr_la(glob_iter, lr_decay)
                return 

            self.model.train()
            for step in range(self.E):
                for x, y, _ in self.trainloader:
                    x, y = x.to(self.device), y.to(self.device)
                
                    self.optimizer.zero_grad()
                    if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                        output=self.model(x)
                        loss=self.ce_loss(output, y)
                    else:
                        output=self.model(x)['output']
                        loss=self.loss(output, y)
                        loss = self.criterion(loss, self.mode)
                    loss.backward()
                    self.optimizer.step()#self.plot_Celeb)
                if lr_decay:
                    if unseen:
                        self.lr_scheduler.step(step)
                    else:
                        self.lr_scheduler.step(glob_iter) #存疑
        else: 
            self.model.train()
            for epoch in range(1, self.local_epochs + 1):
                for i in range(self.K):
                    result =self.get_next_train_batch(count_labels=count_labels)
                    X, y = result['X'], result['y']
                    X, y = X.to(self.device), y.to(self.device)
                    #if count_labels:
                        #self.update_label_counts(result['labels'], result['counts'])

                    self.optimizer.zero_grad()
                    if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                        output=self.model(X)
                        loss=self.ce_loss(output, y)
                    else:
                        output=self.model(X)['output']
                        loss=self.loss(output, y)
                    loss = self.criterion(loss, self.mode)
                    loss.backward()
                    self.optimizer.step()#self.plot_Celeb)
            if lr_decay:
                if unseen:
                    self.lr_scheduler.step(epoch)
                else:
                    self.lr_scheduler.step(glob_iter) #存疑
                
    def criterion(self, loss, mode):
        #self.args.reg > 0 and mode != "PerTrain" and self.args.clients != 1:
        self.m1 = sd_matrixing(self.model.state_dict()).reshape(1, -1).to(self.device)
        self.m2 = sd_matrixing(self.server_param).reshape(1, -1).to(self.device)
        self.m3 = sd_matrixing(self.global_param).reshape(1, -1).to(self.device)
        self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
        self.reg2 = torch.nn.functional.pairwise_distance(self.m1, self.m3, p=2)
        loss = loss + 0.3 * self.reg1 + 0.3 * self.reg2
        return loss

    def set_parameters_SFL(self, global_param, server_param):
        self.model.load_state_dict(global_param)
        self.global_param = global_param
        self.server_param = server_param
    