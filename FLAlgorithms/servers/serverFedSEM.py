from FLAlgorithms.users.userFedSEM import UserFedSEM
from FLAlgorithms.servers.serverbase import *
from FLAlgorithms.trainmodel.models_jiu import Net, SimpleLinear
from utils.model_utils import *
import numpy as np
import matplotlib.pyplot as plt
# Implementation for FedAvg Server
import time
import torch
import torch.nn.functional as F
torch.manual_seed(0)
from tqdm import tqdm

class FedSEM(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        self.p = args.p
        self.models = [create_model_new(args)[0] for p_i in range(self.p)]
        if args.test_unseen == True:
            a = 1
        else:

            for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
                enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
                if train_iterator is None or test_iterator is None:
                    continue
                #GPU_idx = total_users % 6
                #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
                #user_model = create_model_new(args)
                user = UserFedSEM(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating FedSEM server.")

    def flatten(self, source):
        return torch.cat([value.flatten() for value in source.values()])

        
    def get_cluster_idx(self, users): #[tensor(),...tensor()]
        cluster_assign = []
        for i, user in enumerate(users):
            angles = torch.zeros(self.p)
            for j, model_center in enumerate(self.models):
                user_W = {key : value for key, value in user.model.named_parameters()}
                W_j = {key : value for key, value in model_center.named_parameters()}
                s1 = self.flatten(user_W)
                s2 = self.flatten(W_j)
                angles[j] = torch.norm(s1 - s2) #求2范数
            #print(angles)
            #input()
            min_p_i = np.argmin(angles.detach().numpy())
            cluster_assign.append(min_p_i)
        return cluster_assign

    def update_cluster_center(self, cluster_assign, glob_iter, selected_users):
       # for user in selected_users:
            #user.train(glob_iter)  # 用户模型

        local_models = [[] for p in range(self.p)] # 存储每个类对应的用户模型
        for m_i, user in enumerate(selected_users):
            p_i = cluster_assign[m_i] # 第m个用户对应的类
            local_models[p_i].append(user.model) 

        self.timestamp = time.time()
        for p_i, models in enumerate(local_models):
            if len(models) >0:
                self.aggregate_clusterwise(models, self.models[p_i])
        curr_timestamp=time.time() 
        agg_time = curr_timestamp - self.timestamp

        return agg_time 

    def aggregate_clusterwise(self, local_models, global_model): 
        weights = {}

        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)

                weights[name] += param.data

        for name, param in global_model.named_parameters(): # 更新类中心模型
            weights[name] /= len(local_models)
            param.data = weights[name]

    def send_cluster_center(self, beta=1):
        for user in self.users:
            cluster_idx = user.cluster_idx
            #print(cluster_idx)
            user.set_parameters(self.models[cluster_idx],beta=beta)

    def send_cluster_idx(self, cluster_assign, selected_users):
        for i, user in enumerate(selected_users):
            user.set_cluster_idx(cluster_assign[i])
    
    def update_unseen_users(self, unseen_E = 0, unseen_local_epochs = 0):
        #用户预训练
        self.unseen_user_train(unseen_E, unseen_local_epochs)
        #加入离自己最近的模型

        cluster_assign_cur = self.get_cluster_idx(self.unseen_users)
        #self.send_cluster_idx(cluster_assign_cur, self.unseen_users)
        for i, user in enumerate(self.unseen_users):
            cluster_idx = cluster_assign_cur[i]
            user.set_parameters(self.models[cluster_idx])
        '''
        for glob_iter in range(20):
            for user in self.unseen_users:
                user.train_unseen(glob_iter)
        '''
                
    def train(self, args):
        cluster_assign_pre = [0] * args.num_users
        cluster_assign_cur = [0] * args.num_users
        last_glob_iter = self.num_glob_iters # kmeans收敛的glob_iter,初始化为最大值
        
        user_time = 0
        server_time = 0
        train_start= time.time() 

        for glob_iter in range(self.num_glob_iters):
            #cluster_assign = []#每个用户对应的类
            print("-------------Round number: ",glob_iter, " -------------")
            #self.send_p_parameters(self.models) #发送p个模型参数
            self.selected_users = self.select_users(glob_iter,self.num_users)

            self.timestamp = time.time()
            for user in self.selected_users: # allow selected users to train
                user.compute_weight_update(glob_iter, self.personalized)#loss
            curr_timestamp = time.time()
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            user_time += train_time 
            
            self.timestamp = time.time()
            cluster_assign_cur = self.get_cluster_idx(self.selected_users)
            curr_timestamp=time.time() 
            cluster_time = curr_timestamp - self.timestamp

            self.send_cluster_idx(cluster_assign_cur, self.selected_users)
            # 判断收敛需要好几轮类别都不变才行 这个几轮很难把控。。
            # 全部用户参与时可以只判断两轮一致，part of users参与时必须判断好几轮，但实验证明貌似定不了
            # 因为每轮参与的客户端都是随机的
            if cluster_assign_pre == cluster_assign_cur: #下一轮训练时不用判断
                last_glob_iter = glob_iter 
                print(last_glob_iter,"轮:聚类收敛")
                
            
            print(cluster_assign_cur)

            cluster_assign_pre = cluster_assign_cur 
            
            agg_time = self.update_cluster_center(cluster_assign_cur, glob_iter, self.selected_users) #更新类中心模型
            server_total_time = cluster_time + agg_time
            self.metrics['server_agg_time'].append(server_total_time)
            server_time += server_total_time
            
            self.send_cluster_center() # 发送每个用户对应的类中心模型--更新用户模型
            
            self.evaluate()

            self.save_results(args)
            self.save_model_center(args)
            self.save_users_model(args)
            self.save_cluster_assign(args)

        train_end = time.time()
        total_train_time = train_end - train_start
        self.metrics['total_train_time'].append(total_train_time)
        print("user_time = {:.4f}, server_time = {:.4f}, total_time = {:.4f}.".format(user_time, server_time, total_train_time))

        #self.update_unseen_users(unseen_E = 5)
        #self.evaluate_unseen_users()


            
