import torch
import os

from FLAlgorithms.users.userFedU import UserFedU
from FLAlgorithms.servers.serverbase import Server
import numpy as np
# Implementation for FedAvg Server
import time
from tqdm import tqdm
import torch
from utils.model_utils import *

# Implementation for FedAvg Server

class FedU(Server):
    def __init__(self, args, model, data_participate, data_unseen, A, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)
        # Initialize data for all  users
        #subset data
        
        self.L_k = args.L_K
        self.K = args.K
        N=self.num_users
        b = np.random.uniform(0,1,size=(N,N))
        b_symm = (b + b.T)/2 #对称矩阵
        b_symm[b_symm < 0.25] = 0 #
        self.alk_connection = b_symm
        self.A = A

        #np.random.seed(0)
        if args.test_unseen == True:
            a = 1
        else:
            for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
                enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
                if train_iterator is None or test_iterator is None:
                    continue
                #GPU_idx = task_id % 6
                #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
                #user_model = create_model_new(args, user_device)
                user = UserFedU(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes,  use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples


            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating FedU server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self, args):
        loss = []
        # only board cast one time
        self.send_parameters(self.users, mode=self.mode)
        #self.meta_split_users()
        for glob_iter in range(self.num_glob_iters):
            
            print("-------------Round number: ",glob_iter, " -------------")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            # local update at each users
            for user in self.selected_users:
                user.train(glob_iter, personalized=self.personalized) #* user.train_samples
                
            # Agegrate parameter at each user
            if(self.L_k != 0): # if L_K = 0 it is local model 
                for user in self.selected_users:
                    user.aggregate_parameters(self.selected_users, glob_iter, self.num_users , self.A)
            self.evaluate()
            #self.meta_evaluate()
            self.save_results(args)
            self.save_model(args)
            self.save_users_model(args)