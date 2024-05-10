import torch
from FLAlgorithms.users.userbase import User
import numpy as np
import copy

class UserIFCA(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, p, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes,  use_adam=False)
        self.p = p
        self.cluster_idx = 0
        self.losses ={}
        self.models= [] # 存储从server收到的p个模型
        
    def set_p_parameters(self, models):
        for p_i, model in enumerate(models):
            self.models.append(model)
        
    def get_cluster_idx(self):
        corrects = {}
        for p_i in range(self.p):
            loss = 0
            with torch.no_grad():
                for x, y,_ in self.trainloader: 
                    x, y = x.to(self.device), y.to(self.device)
                    if 'cifar10' in self.dataset or 'shakespeare' in self.dataset or 'cifar100' in self.dataset:   
                        output = self.models[p_i](x)
                        loss += self.ce_loss(output, y)
                    else:
                        output = self.models[p_i](x)['output']
                        loss += self.loss(output, y)
                    #n_correct = self.n_correct(y_logits, y)
    
                    self.losses[p_i] = loss.item()
            
        machine_losses = [ self.losses[p_i] for p_i in range(self.p) ]
        min_p_i = np.argmin(machine_losses)
        self.cluster_idx = min_p_i

        return self.cluster_idx


    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.model = copy.deepcopy(self.models[self.cluster_idx]) # 将用户模型参数设置为对应类别的模型参数
        #self.models[self.cluster_idx].to(device)
        for old_param, new_param in zip(self.model.parameters(), self.models[self.cluster_idx].parameters()):
                old_param.data = new_param.data.clone()

        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)



    


   

