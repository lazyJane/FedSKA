from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
import numpy as np
# Implementation for FedAvg Server
import time
from tqdm import tqdm
import torch
from utils.model_utils import *
class FedAvg(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)
        
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
                user = UserAVG(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes, use_adam=False)

                self.users.append(user)
                self.total_train_samples += user.train_samples


            
            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating FedAvg server.")

    def train(self, args):
        train_start= time.time() 
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            
            
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.send_parameters(self.users, mode=self.mode)
            self.evaluate()


            self.timestamp = time.time() 
            for user in self.selected_users: # allow selected users to train
                user.train(glob_iter, personalized=self.personalized) #* user.train_samples
            curr_timestamp = time.time()
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            # Evaluate selected user
            if self.personalized:
                # Evaluate personal model on user for each iteration
                print("Evaluate personal model\n")
                self.evaluate_personalized_model()

            self.timestamp = time.time()
            self.aggregate_parameters()
            curr_timestamp=time.time() 
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            
            self.save_results(args)
            self.save_model(args)
            self.save_users_model(args)


        train_end = time.time()
        total_train_time = train_end - train_start
        self.metrics['total_train_time'].append(total_train_time)

        #self.update_unseen_users()
        #self.evaluate_unseen_users()
