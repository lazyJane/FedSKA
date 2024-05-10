import torch
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import FedProxOptimizer,ProxSGD

class UserFedDS(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes,  use_adam=False)
        self.W = {key : value for key, value in self.model.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.cluster_idx = 0
        #self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.1, momentum=0.9)
        
    
    def set_cluster_idx(self, cluster_idx):
        self.cluster_idx = cluster_idx
        
    def copy(self, target, source):
        for name in target:
            target[name].data = source[name].data.clone()

    def subtract_(self, target, minuend, subtrahend):
        for name in target:
            target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

    def reset(self):
        self.copy(target=self.W, source=self.W_old)

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        #self.model = copy.deepcopy()
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)


    def compute_weight_update(self, glob_iter, personalized):
        self.copy(target=self.W_old, source=self.W)
        #self.optimizer.param_groups[0]["lr"]*=0.99
        self.train(glob_iter) 
        self.subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)  

    