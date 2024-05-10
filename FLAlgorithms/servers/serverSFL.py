from FLAlgorithms.users.userSFL import UserSFL
from FLAlgorithms.servers.serverbase import Server
import numpy as np
# Implementation for FedAvg Server
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.model_utils import *
from GraphConstructor import GraphConstructor
import copy
class FedSFL(Server):
    def __init__(self, args, model, data_participate, data_unseen, A, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)
        
        self.A = A
        # initialization 
        w_original = self.model.state_dict()
        self.w_server = [w_original] * args.num_users # w_server: w_i
        self.personalized_model_u = copy.deepcopy(self.w_server) # personalized_model_u: u_i
        
        if args.test_unseen == True:
            a = 1
        else:
            for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
                enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
                if train_iterator is None or test_iterator is None:
                    continue
                user = UserSFL(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes,  use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            
            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating FedSFL server.")

    def train(self, args):
        train_start= time.time() 
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
        
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.send_parameters_SFL(self.users)
            

            self.timestamp = time.time() 
            for user in self.selected_users: # allow selected users to train
                user.train(glob_iter, personalized=self.personalized) #* user.train_samples
                self.w_server[user.id] = user.model.state_dict() #updated w models

            curr_timestamp = time.time()
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            # Evaluate selected user
            if self.personalized:
                # Evaluate personal model on user for each iteration
                print("Evaluate personal model\n")
                self.evaluate_personalized_model()

            self.timestamp = time.time()
            
            self.personalized_model_u = self.graph_dic(self.w_server, self.A, args) #updated personalized_u models
            self.read_out(self.personalized_model_u, args.device) # w_t

            
            curr_timestamp=time.time() 
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            
            self.evaluate() # 

            self.save_results(args)
            self.save_model(args)
            self.save_users_model(args)


        train_end = time.time()
        total_train_time = train_end - train_start
        self.metrics['total_train_time'].append(total_train_time)

        #self.update_unseen_users()
        #self.evaluate_unseen_users()

    def send_parameters_SFL(self, users):
        for user in self.selected_users:
            global_param = self.model.state_dict()
            server_param = self.personalized_model_u[user.id]
            user.set_parameters_SFL(global_param, server_param)

    def read_out(self, personalized_models, device):
        # average pooling as read out function
        global_model = self.average_dic(personalized_models, device, 0)
        self.model.load_state_dict(global_model)
        #return [global_model] * len(personalized_models)

    def average_dic(self, model_dic, device, dp=0.001):
        w_avg = copy.deepcopy(model_dic[0])
        for k in w_avg.keys():
            for i in range(1, len(model_dic)):
                w_avg[k] = w_avg[k].data.clone().detach() + model_dic[i][k].data.clone().detach()
            #print(torch.mul(torch.randn(w_avg[k].shape), dp).device)
            w_avg[k] = w_avg[k].data.clone().detach().div(len(model_dic)) + torch.mul(torch.randn(w_avg[k].shape), dp).to(self.device)
        return w_avg

    def graph_dic(self, models_dic, pre_A, args):
        '''
        models_dic = w_server = [w_server] * args.num_users   [updated local model1, updated local model2,...]
        '''
        keys = []
        key_shapes = []
        param_metrix = []

        for model in models_dic:
            param_metrix.append(sd_matrixing(model, aggregate = True).clone().detach())
        param_metrix = torch.stack(param_metrix) # [tensor([1, 2]), tensor([1, 2]), tensor([1, 2])] -> tensor([[1, 2],[1, 2],[1, 2]])

        for key, param in models_dic[0].items():
            keys.append(key)
            key_shapes.append(list(param.data.shape))

        if args.agg == "graph_v2" or args.agg == "graph_v3":
            # constract adj
            
            k = min(args.k, args.num_users)
            A = self.generate_adj(param_metrix, args, k).cpu().detach().numpy()
            A = self.normalize_adj(A)
            A = torch.tensor(A)
            if args.agg == "graph_v3":
                pre_A = torch.tensor(pre_A)
                A = (1 - args.adjbeta) * pre_A + args.adjbeta * A
        else:
            A = pre_A

        #print("A",A)
        #input()
        param_metrix = []
        for model in models_dic:
            param_metrix.append(sd_matrixing(model,aggregate=True).clone().detach())
        param_metrix = torch.stack(param_metrix)
        # Aggregating
        A = A.to(self.device)
        self.metrics['adj'].append(A.detach().cpu().numpy())

        '''
        param_metrix: [model1_all_parameters_flatten()[],model2[],]
        '''
        aggregated_param = torch.mm(A, param_metrix)
        # print(aggregated_param.shape) # torch.Size([80, 7850])
        for i in range(args.gen_layers - 1):
            aggregated_param = torch.mm(A, aggregated_param) #u
        #print(aggregated_param)
        new_param_matrix = (args.serveralpha * aggregated_param) + ((1 - args.serveralpha) * param_metrix)

        '''
        keys: list() [model.layer_name1,...]
        key_shapes:list() [shape1,shape2,...]
        '''
        # reconstract parameter
        for i in range(len(models_dic)):
            pointer = 0
            for k in range(len(keys)):
                num_p = 1
                for n in key_shapes[k]:
                    num_p *= n
                models_dic[i][keys[k]] = new_param_matrix[i][pointer:pointer + num_p].reshape(key_shapes[k])
                pointer += num_p

        return models_dic
    
    def get_0_1_array(self, array,rate=0.2):
        '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
        zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
        new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
        new_array[:zeros_num] = 1 #将一部分换为0
        np.random.shuffle(new_array)#将0和1的顺序打乱
        re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
        return re_array


    def prepare_A(self):
        edge_frac = 0.5
        self.A = self.get_0_1_array(self.A, edge_frac)
        def RBF(x, L, sigma):

            #x: 待分类的点的坐标 
            #x': 中心点，通过计算x到x'的距离的和来判断类别
            #sigma：有效半径

            return np.exp(-(np.sum((x - L) ** 2)) / (2 * sigma**2)) 
    
    def generate_adj(self, param_metrix, args, k):
        dist_metrix = torch.zeros((len(param_metrix), len(param_metrix)))
        for i in range(len(param_metrix)):
            for j in range(len(param_metrix)):
                #cosine = nn.CosineSimilarity(dim=1)
                #dist_metrix[i][j] = cosine(param_metrix[i].view(1, -1), param_metrix[j].view(1, -1))
                dist_metrix[i][j] = torch.nn.functional.pairwise_distance(
                    param_metrix[i].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach() # l2 distance
        
        dist_metrix = torch.nn.functional.normalize(dist_metrix).to(args.device)
        #print("dist_metrix", dist_metrix) 
        
        gc = GraphConstructor(args.num_users, k, args.node_dim,
                            args.device, args.adjalpha).to(args.device)
        idx = torch.arange(args.num_users).to(args.device)
        optimizer = torch.optim.SGD(gc.parameters(), lr=args.learning_rate, weight_decay=0.0001)

        for e in range(args.epochs_adj):
            optimizer.zero_grad()
            adj = gc(idx) # idx [0,1,2,...]
            adj = torch.nn.functional.normalize(adj) #归一化

            loss = torch.nn.functional.mse_loss(adj, dist_metrix)
            loss.backward()
            optimizer.step()

        adj = gc.eval(idx)

        return adj
    
    def normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx