import torch
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import FedProxOptimizer,ProxSGD

class UserFedProx(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes,  use_adam=False)

        if 'mnist' == self.dataset[0]:
            self.optimizer = ProxSGD([param for param in self.model.parameters() if param.requires_grad], lr=self.learning_rate, mu=0.01)
        else: 
            self.optimizer = ProxSGD([param for param in self.model.parameters() if param.requires_grad], lr=self.learning_rate, mu=0.01,momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        
    def train(self, glob_iter, lr_decay=True, count_labels=False):
        # cache global model initialized value to local model
        self.clone_model_paramenter(self.local_model, self.model.parameters())
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)
        
     