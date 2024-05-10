import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import *
import numpy as np
from tqdm import tqdm
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
            enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
            if train_iterator is None or test_iterator is None:
                continue
            #GPU_idx = total_users % 6
            #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
            user = UserpFedMe(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes,  use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",args.num_users, " / " , self.total_users)
        print("Finished creating pFedMe server.")

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
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters(self.users, mode=self.mode)

            # Evaluate gloal model on user for each interation
            #print("Evaluate global model")
            #print("")
            #self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.personalized_aggregate_parameters()
            #self.evaluate()


        #print(loss)
            self.save_results(args)
        self.save_model(args)
    
  
