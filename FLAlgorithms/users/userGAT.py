from fcntl import F_SETLEASE
import torch
from FLAlgorithms.users.userbase import User
import copy
from utils.model_utils import *

class UserGAT(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False)
        self.global_param = self.model.state_dict()
        self.server_param = self.model.state_dict()
        self.server_param_self = self.model.state_dict()
        
        self.mode = "Train"


    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        unseen = False
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)
                
    def criterion(self, loss, mode):
        #self.args.reg > 0 and mode != "PerTrain" and self.args.clients != 1:
        self.m1 = sd_matrixing(self.model.state_dict()).reshape(1, -1).to(self.device)
        self.m2 = sd_matrixing(self.server_param).reshape(1, -1).to(self.device)
        self.m3 = sd_matrixing(self.global_param).reshape(1, -1).to(self.device)
        self.m4 = sd_matrixing(self.server_param_self).reshape(1, -1).to(self.device)
        self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
        self.reg2 = torch.nn.functional.pairwise_distance(self.m1, self.m3, p=2)
        self.reg3 = torch.nn.functional.pairwise_distance(self.m1, self.m4, p=2)
        #loss = loss + 0.3 * self.reg1 + 0.3 * self.reg2  + 0.3  * self.reg3
        loss = loss + 0.3 * self.reg1 + 0.3 * self.reg2 
        return loss

    def set_parameters_SFL(self, global_param, server_param, server_param_self):
        self.model.load_state_dict(global_param)
        self.global_param = global_param
        self.server_param = server_param
        self.server_param_self = server_param_self

    