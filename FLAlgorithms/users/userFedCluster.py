import torch
from FLAlgorithms.users.userbase import User

class UserFedCluster(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, n_classes, use_adam=use_adam)

        self.W = {key : value for key, value in self.model.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        #self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.1, momentum=0.9)

    def copy(self, target, source):
        for name in target:
            target[name].data = source[name].data.clone()

    def subtract_(self, target, minuend, subtrahend):
        for name in target:
            target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

    def reset(self):
        self.copy(target=self.W, source=self.W_old)

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)


    def compute_weight_update(self, glob_iter, personalized):
        self.copy(target=self.W_old, source=self.W)
        #self.optimizer.param_groups[0]["lr"]*=0.99
        self.train(glob_iter, personalized) 
        self.subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)  

    