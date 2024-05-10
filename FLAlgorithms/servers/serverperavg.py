import torch
import os
import numpy as np
from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import *

# Implementation for per-FedAvg Server
from tqdm import tqdm
class PerAvg(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        if args.test_unseen == True:
            a = 1
        else:
            for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
                enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
                if train_iterator is None or test_iterator is None:
                    continue
                #GPU_idx = total_users % 6
                #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
                user = UserPerAvg(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes,  use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating PerFedAvg server.")

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

    def evaluate_one_step(self, save=True):
        for c in self.users:
            c.train_one_step()

        self.evaluate()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

    def train(self, args):
        loss = []
        train_start= time.time() 
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            if glob_iter == 0: self.send_parameters(self.users, mode=self.mode)

            # Evaluate gloal model on user for each interation
            #print("Evaluate global model with one step update")
            #print("")
            #self.evaluate_one_step()

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
                
            self.aggregate_parameters()
            self.send_parameters(self.users, mode=self.mode)
            #self.evaluate()
            self.evaluate_one_step()

            self.save_results(args)
            self.save_model(args)
            self.save_users_model(args)

        train_end = time.time()
        total_train_time = train_end - train_start
        self.metrics['total_train_time'].append(total_train_time)

        #self.update_unseen_users(unseen_E=5)
        #self.evaluate_unseen_users()
