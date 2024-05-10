from FLAlgorithms.users.userFedCluster import UserFedCluster
from FLAlgorithms.servers.serverbase import Server
import numpy as np
# Implementation for FedCluster Server
import time
import torch 
#from sklearn.cluster import AgglomerativeClustering
from utils.helper import ExperimentLogger, display_train_stats
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
class FedCluster(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)
        
    
        for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
            enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
            if train_iterator is None or test_iterator is None:
                continue
            #GPU_idx = total_users % 6
            #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
            user = UserFedCluster(args, task_id, model, train_iterator, val_iterator, test_iterator, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",args.num_users, " / " , total_users)
        print("Finished creating FedAvg server.")

        self.cluster_indices = [np.arange(self.num_users).astype("int")] 
        self.user_clusters = [[self.users[i] for i in idcs] for idcs in self.cluster_indices]
        #self.EPS_1 = 0.06
        #self.EPS_2 = 0.1
        self.EPS_1 = 0.4
        self.EPS_2 = 1.6 #

    def flatten(self, source):
            return torch.cat([value.flatten() for value in source.values()]) #dim默认为0

    def pairwise_angles(self, sources):
            angles = torch.zeros([len(sources), len(sources)])
            for i, source1 in enumerate(sources):
                 for j, source2 in enumerate(sources):
                    s1 = self.flatten(source1)
                    s2 = self.flatten(source2)
                    angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)
            return angles.numpy()

    def compute_pairwise_similarities(self, users):
            return self.pairwise_angles([user.dW for user in users])
        
    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(self.flatten(user.dW)).item() for user in cluster])
    
    def compute_weighted_mean_update_norm(self, cluster):
        result = 0
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples # 所有用户样本总数
        for user in cluster:
            result += self.flatten(user.dW) * user.train_samples / total_train
        result = torch.norm(result).item()
        return result

    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([self.flatten(user.dW) for user in cluster]), 
                                     dim=0)).item()

    def cluster_users(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten() 
        c2 = np.argwhere(clustering.labels_ == 1).flatten() 
        return c1, c2

    '''
    def reduce_add_average(self, cluster, targets, sources):#sources=[{key:values},{key:values},...{key:values}]
        total_train = 0
        for user in cluster:
            total_train += user.train_samples # 所有用户样本总数
        ratio= [0 for _ in range(len(cluster))]  
        for i, user in enumerate(cluster):
            ratio[i] = user.train_samples / total_train
        for target in targets:
            for name in target:
                tmp = 0
                for i, source in enumerate(sources):
                    tmp += source[name].data * ratio[i] #dW1*ratio1+dW2*ratio*2+...
                target[name].data += tmp
    '''

    def reduce_add_average(self, targets, sources):
        for target in targets:
            for name in target:
                tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone() #所有用户模型参数的均值
                target[name].data += tmp

    def aggregate_clusterwise(self, user_clusters):
        for cluster in user_clusters:
            self.reduce_add_average(targets=[user.W for user in cluster], 
                               sources=[user.dW for user in cluster], )


    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            if glob_iter == 1:
                self.send_parameters(mode=self.mode)
            self.evaluate() # 每个用户在自己的测试集上测试准确率的均值 

            self.timestamp = time.time() # log user-training + compute dW + similarities start time train+计算dW时间
            for user in self.selected_users: # allow selected users to train
                user.compute_weight_update(glob_iter, self.personalized)#loss
                user.reset() # 之后aggregate_clusterwise是根据dW做聚合
            similarities = self.compute_pairwise_similarities(self.selected_users)##shape[10,10] 计算相似度
            curr_timestamp = time.time() # log user-training + compute dW + similarities start time
            compute_cos_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            #self.metrics['compute_cos_time'].append(compute_cos_time)
            #print('compute_cos_time',compute_cos_time)
                
            cluster_indices_new = []
            for idc in self.cluster_indices: #idc:[1,2,...,10]  对每个类
                max_norm = self.compute_max_update_norm([self.users[i] for i in idc])#类中所有用户dW的最大值
                mean_norm = self.compute_mean_update_norm([self.users[i] for i in idc])#这一类的平均dW
                #weighted_norm = self.compute_weighted_mean_update_norm([self.users[i] for i in idc])
                print('max_norm',max_norm)
                print('mean_norm',mean_norm)
                
                if mean_norm<self.EPS_1 and max_norm>self.EPS_2 and len(idc)>4 and glob_iter>50: #0.4, 1.6
                    print("执行了一次聚类")
                    #server.cache_model(idc, users[idc[0]].W, acc_users) 
                    c1, c2 = self.cluster_users(similarities[idc][:,idc]) #[[1,2,...,10]][:,[1,2,...,10]]取这一类的相似度矩阵

                    cluster_indices_new += [c1, c2]
                    print(cluster_indices_new)

                    print("split",glob_iter)

                else:
                    cluster_indices_new += [idc]

            self.cluster_indices = cluster_indices_new # 新的类划分
            self.user_clusters = [[self.users[i] for i in idcs] for idcs in self.cluster_indices]

            self.timestamp = time.time() # log server-agg start time
            self.aggregate_clusterwise(self.user_clusters)
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            #print('server_agg_time',agg_time)
            

            self.save_results(args)
            #self.save_model()