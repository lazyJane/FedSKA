from FLAlgorithms.users.userlocal import UserLocal
from FLAlgorithms.servers.serverbase import *
import torch
import torch.nn.functional as F
#from utils import public_data
import numpy as np
# Implementation for FedAvg Server
import time
from tqdm import tqdm
import pandas as pd
class Fedlocal(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)
    
        for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
            enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
            if train_iterator is None or test_iterator is None:
                continue
            #GPU_idx = total_users % 6
            #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
            user = UserLocal(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",args.num_users, " / " , self.total_users)
        print("Finished creating Fedlocal server.")

    def get_logits_clients(self, selected_users):
        logits = [] 
        for i, user in enumerate(selected_users):
             with torch.no_grad():
                for x, y,_ in self.public_loader: 
                    x, y = x.to(self.device), y.to(self.device)
                    if 'cifar' in self.dataset: 
                        logits.append(user.model(x))
                    else:
                        logits.append(user.model(x)['logit']) #[tensor(),...tensor()]
        return logits
        
    def compute_pairwise_KL(self, sources):
            angles = torch.zeros([len(sources), len(sources)])
            for i, source1 in enumerate(sources):
                 for j, source2 in enumerate(sources):
                    angles[i,j] = F.kl_div(F.log_softmax(source1, dim=1), F.softmax(source2, dim=1))
            return angles.numpy()

    def compute_pairwise_JS(self, sources):
            angles = torch.zeros([len(sources), len(sources)])
            for i, source1 in enumerate(sources):
                 for j, source2 in enumerate(sources):
                    p_output = F.softmax(source1, dim=1)
                    q_output = F.softmax(source2, dim=1)
                    log_mean_output = ((p_output + q_output )/2).log()
                    angles[i,j] = (F.kl_div(log_mean_output, p_output) + F.kl_div(log_mean_output, q_output)) / 2
            data = pd.DataFrame(angles.numpy())

            writer = pd.ExcelWriter('./similarity matrix.xlsx')		# 写入Excel文件
            data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
            writer.save()

            return angles.numpy()
    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            #self.send_parameters(mode=self.mode)
            
            for user in self.selected_users: # allow selected users to train
                user.train(glob_iter, personalized=self.personalized) #* user.train_samples
            
            # Evaluate selected user
            self.evaluate()
            
            self.save_results(args)
            self.save_model(args)
            self.save_users_model(args)
        logits = self.get_logits_clients(self.users)
        print(logits)
        similarities = self.compute_pairwise_JS(logits)##shape[10,10] 计算相似度