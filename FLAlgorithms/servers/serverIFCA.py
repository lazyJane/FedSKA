from FLAlgorithms.users.userIFCA import UserIFCA
from FLAlgorithms.servers.serverbase import *
from FLAlgorithms.trainmodel.models_jiu import Net, SimpleLinear
import numpy as np
import matplotlib.pyplot as plt
# Implementation for FedAvg Server
import time
torch.manual_seed(0)
from tqdm import tqdm
from utils.model_utils import *

class FedIFCA(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        self.p = args.p
        if args.load_model == True:
                self.models = self.load_center_model()
        else: self.models = [create_model_new(args)[0] for p_i in range(self.p)]
        if args.test_unseen == True:
            a = 1
        else:
            for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
                enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
                if train_iterator is None or test_iterator is None:
                    continue
                #GPU_idx = total_users % 6
                #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
                user = UserIFCA(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.p, self.n_classes, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating FedICFA server.")


    def update_cluster_center(self, cluster_assign, glob_iter, selected_users):
        user_train_start = time.time()
        for user in selected_users:
            user.train(glob_iter)  # 用户模型 用此轮所属类别的模型训练
        user_train_end = time.time()
        user_train = user_train_end - user_train_start

        local_models = [[] for p in range(self.p)]
        for m_i, user in enumerate(selected_users):
            p_i = cluster_assign[m_i]
            local_models[p_i].append(user.model)

        self.timestamp = time.time()
        for p_i, models in enumerate(local_models):
            if len(models) >0:
                self.aggregate_clusterwise(models, self.models[p_i])
        curr_timestamp=time.time() 
        agg_time = curr_timestamp - self.timestamp
        self.metrics['server_agg_time'].append(agg_time)

        return user_train, agg_time
            

    def aggregate_clusterwise(self, local_models, global_model):
        weights = {}
        #global_model.to(self.device)
        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                #print(name)
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)

                weights[name] += param.data
            #input()

        for name, param in global_model.named_parameters():
            weights[name] /= len(local_models)
            param.data = weights[name]

    def get_cluster(self, selected_users):
        cluster_assign = []
        for user in selected_users:
            cluster_assign.append(user.get_cluster_idx()) #根据发送的模型参数确定类别

        return cluster_assign
    
    def send_cluster_center(self, users, beta=1):
        for user in users:
            cluster_idx = user.cluster_idx
            user.set_parameters(self.models[cluster_idx],beta=beta)

    def update_unseen_users(self, unseen_E = 0, unseen_local_epochs = 0):
        #用户预训练
        #self.unseen_user_train(unseen_E)
        #加入离自己最近的模型
        for user in self.unseen_users:
            user.set_p_parameters(self.models)
        cluster_assign_unseen = self.get_cluster(self.unseen_users)
        for i, user in enumerate(self.unseen_users):
            cluster_idx = cluster_assign_unseen[i]
            user.set_parameters(self.models[cluster_idx])
            
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
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.send_p_parameters(self.models) #发送p个模型参数 
            self.evaluate()
            
            #if glob_iter <= last_glob_iter:
            user_get_cluster_idx_time_start = time.time() 
            cluster_assign_cur = self.get_cluster(self.selected_users) #得到类别
            user_get_cluster_idx_time_end = time.time() 
            user_get_cluster_idx_time = user_get_cluster_idx_time_end - user_get_cluster_idx_time_start
            if cluster_assign_pre == cluster_assign_cur: #下一轮训练时不用判断
                last_glob_iter = glob_iter 
                print(last_glob_iter,"轮:聚类收敛")
            #else:
                #self.send_cluster_center(self.users)

            print(cluster_assign_cur)
            cluster_assign_pre = cluster_assign_cur 

            user_time, agg_time = self.update_cluster_center(cluster_assign_cur, glob_iter, self.selected_users) #更新类中心 并且用户已经用最优loss的模型训练过了
            user_total_time = user_get_cluster_idx_time + user_time
            self.metrics['user_train_time'].append(user_total_time)
            user_time += user_total_time 
            server_time += agg_time

            
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


    