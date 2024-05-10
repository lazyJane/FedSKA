from sklearn.metrics import roc_auc_score
from FLAlgorithms.users.userSKA import userSKA 
from FLAlgorithms.users.userlocal import UserLocal
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.gnn import *
from FLAlgorithms.trainmodel.gnn_IDGL import *
from FLAlgorithms.trainmodel.graphlearn import *
import itertools
import numpy as np

np.set_printoptions(threshold=np.inf)
# Implementation for FedAvg Server
import time
from tqdm import tqdm
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
from utils.model_utils import *
from utils.graph_utils import *
from collections import defaultdict

import copy
import dgl
from utils.datasets import *

class FedSKA_shakespeare(Server):
    def __init__(self, args, model, data_participate, data_unseen, A, g, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)
        
        #print("len(self.train_iterators)",len(self.train_iterators))
        if args.test_unseen == True:
            a = 1
        else:
            self.server_train_dataset_x = []
            self.server_train_dataset_y = []
            self.server_test_dataset_x = []
            self.server_test_dataset_y = []
            self.train_max_len = 0
            self.test_max_len = 0
            self.train_min_len = INF
            self.test_min_len = INF
            for task_id, (train_dataset, test_dataset, train_iterator, val_iterator, test_iterator, len_train, len_test) in \
                enumerate(tqdm(zip(self.train_datasets, self.test_datasets, self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
                
                if train_iterator is None or test_iterator is None:
                    continue
                user = userSKA(args, task_id, model, train_dataset, test_dataset, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes, use_adam=False)
                #user = UserLocal(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes, use_adam=False)
                self.users.append(user)
                #print(train_dataset.data.type())
                 
                if 'shakespeare' in self.dataset:
                    
                    train_x = train_dataset.data 
                    #print(train_x.shape[0])
                    self.train_max_len = max(self.train_max_len, train_x.shape[0])
                    self.train_min_len = min(self.train_min_len, train_x.shape[0])
                    #print(self.train_min_len)

                    test_x = test_dataset.data 
                    self.test_max_len = max(self.test_max_len, test_x.shape[0])
                    self.test_min_len = min(self.test_min_len, test_x.shape[0])

                
                self.total_train_samples += user.train_samples

            if 'shakespeare' in self.dataset:
                for task_id, (train_dataset, test_dataset, train_iterator, test_iterator) in \
                enumerate(tqdm(zip(self.train_datasets, self.test_datasets, self.train_iterators, self.test_iterators), total=len(self.train_iterators))):
            
                        if train_iterator is None or test_iterator is None:
                            continue
                        #print(self.train_max_len) #24942
                        train_x = train_dataset.data 
                        train_y = train_dataset.targets
                        test_x = test_dataset.data #torch.Size([119, 32, 32, 3])
                        test_y = test_dataset.targets
                        # train_x shape [len,seq_len]    

                        if train_x.shape[0] > self.train_min_len:
                            train_x = train_x[:self.train_min_len]
                            train_y = train_y[:self.train_min_len]
                        if test_x.shape[0] > self.test_min_len:
                            test_x = test_x[:self.test_min_len]
                            test_y = test_y[:self.test_min_len]
                        #print(train_x.shape)
                        #print(test_x.shape)
                     
                        '''
                        if train_x.shape[0] < self.train_max_len:
                            
                            #print(train_x.shape)#torch.Size([760, 80])
                            train_x_buff = torch.zeros(self.train_max_len, train_x.shape[1])
                            train_y_buff = torch.zeros(self.train_max_len, train_y.shape[1])
                            train_x_buff[:train_x.shape[0],:] = train_x
                            #print(train_x[-1].repeat(self.train_max_len-train_x.shape[0],1).shape)
                            train_x_buff[train_x.shape[0]:,:] = train_x[-1].repeat(self.train_max_len-train_x.shape[0],1)
                            train_y_buff[:train_y.shape[0],:] = train_y
                            train_y_buff[train_y.shape[0]:,:] = train_y[-1].repeat(self.train_max_len-train_y.shape[0],1)
                            train_x = train_x_buff.long()
                            train_y = train_y_buff.long()
                        
                        
                        if test_x.shape[0] < self.test_max_len:
                            
                            test_x_buff = torch.zeros(self.test_max_len, test_x.shape[1])
                            test_y_buff = torch.zeros(self.test_max_len, test_y.shape[1])
                            test_x_buff[: test_x.shape[0],:] = test_x
                            test_x_buff[test_x.shape[0]:,:] = test_x[-1].repeat(self.test_max_len-test_x.shape[0],1)
                            test_y_buff[:test_y.shape[0],:] = test_y
                            test_y_buff[test_y.shape[0]:,:] = test_y[-1].repeat(self.test_max_len-test_y.shape[0],1)
                            test_x = test_x_buff.long()
                            test_y = test_y_buff.long()
                        '''
                        self.server_train_dataset_x.append(train_x) 
                        self.server_train_dataset_y.append(train_y)
                        self.server_test_dataset_x.append(test_x)
                        self.server_test_dataset_y.append(test_y)
            
            self.init_server_dataset(args)
            
            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating FedSKA server.")

        # initialization graph
        self.original_g = g.to(self.device)
        #self.original_A = torch.tensor(A).to(self.device)
        if args.sparse:
            self.original_A = sparse_mx_to_torch_sparse_tensor(A) 
        else:
            self.original_A = torch.tensor(A).to(self.device)
        self.init_graph(args)
        
        w_original = self.model.state_dict()
        self.w_server = [w_original] * args.num_users # w_server: w_i
        self.personalized_model_u = copy.deepcopy(self.w_server) # personalized_model_u: u_i
    
        self.init_loss_fn()
        
    def init_graph(self, args):
        self.server_epoch = args.server_epoch
        
        # initialize graph generator
        for user in self.users: # allow selected users to train
            user.train(glob_iter=0, personalized=self.personalized) #* user.train_samples
        user_embeddings = self.get_user_embeddings(args, users=self.users).to(self.device)
        nfeat = user_embeddings.shape[-1]
        args.mlp_h = nfeat 
        # define learn_adj_model
        self.graph_model_SLAPS = GCN_DAE(nlayers=args.nlayers_adj, 
                            in_dim=nfeat, 
                            hidden_dim=args.hidden_adj, 
                            nclasses=nfeat,
                            dropout=args.dropout1, 
                            dropout_adj=args.dropout_adj1,
                            features = user_embeddings, 
                            k=args.k, 
                            knn_metric = args.knn_metric, 
                            i=args.i, 
                            non_linearity=args.non_linearity,
                            normalization=args.normalization,
                            mlp_h=args.mlp_h, 
                            mlp_epochs=args.mlp_epochs,
                            mlp_act=args.mlp_act,
                            feature_hidden_size=nfeat, 
                            graph_learn_hidden_size = args.graph_learn_hidden_size, 
                            graph_learn_epsilon = args.graph_learn_epsilon, 
                            graph_learn_num_pers = args.graph_learn_num_pers, 
                            graph_metric_type = args.graph_metric_type, 
                            gen_mode = args.gen_mode, 
                            sparse=args.sparse,
                            device=self.device).to(self.device)
        
        self.optimizer_graph_model_SLAPS = torch.optim.Adam(self.graph_model_SLAPS.parameters(), lr=args.learning_rate, weight_decay=args.w_decay_adj)
        
        # initialize unsupervised learning objective
        # 这里用GCN_C最好啊！ 因为我这个虽然不是slaps中的train_two_steps,即没有用的客户端的所属类别标签信息，但是每一轮算上来的adj都要变化，这个GCN是为了计算新的graph_based_user_feature
        self.gcn = GCN_C(in_channels=SHAKESPEARE_CONFIG["hidden_size"], hidden_channels=SHAKESPEARE_CONFIG["hidden_size"], out_channels=SHAKESPEARE_CONFIG["hidden_size"],num_layers=args.gen_layers,
                       dropout=args.dropout2, dropout_adj=args.dropout_adj2, sparse=args.sparse).to(self.device)
        self.server_optimizer_gcn = torch.optim.Adam(params=self.gcn.parameters(), lr=self.learning_rate, weight_decay=args.server_weight_decay)

    def init_server_dataset(self, args):
        if 'metr-la' in self.dataset:
            self.server_train_dataloader = self.public_loader[0] 
            self.server_test_dataloader = self.public_loader[1] 
        elif 'shakespeare' in self.dataset:
            ###torch.Size([100, L, 28, 28])#N,B,W,H->B,N,W,H ([951, 80]) #[B,seq_len]
            #print(self.server_train_dataset_x.shape) #[L,seq_len]   # N,L,seq_len  
            self.server_train_dataset_x = torch.stack(self.server_train_dataset_x).permute(1,0,2)  # N,L,seq_len >L,N,seq_len
            self.server_train_dataset_y = torch.stack(self.server_train_dataset_y).permute(1,0,2)  
            self.server_train_dataset = SnapShotDataset(self.server_train_dataset_x, self.server_train_dataset_y)
            self.server_train_dataloader = list(DataLoader(self.server_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))

            self.server_test_dataset_x = torch.stack(self.server_test_dataset_x).permute(1,0,2)
            self.server_test_dataset_y = torch.stack(self.server_test_dataset_y).permute(1,0,2)
            self.server_test_dataset = SnapShotDataset(self.server_test_dataset_x, self.server_test_dataset_y)
            self.server_test_dataloader = list(DataLoader(self.server_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True))
        elif 'mnist' in self.dataset:
            self.server_train_dataset_x = torch.stack(self.server_train_dataset_x).permute(1, 0, 2, 3)###torch.Size([100, 520, 28, 28])#N,B,W,H->B,N,W,H
            self.server_train_dataset_y = torch.stack(self.server_train_dataset_y).permute(1, 0) ##torch.Size([100, 520])#N,B->#B,N 
            self.server_train_dataset = ServerDataset(self.server_train_dataset_x, self.server_train_dataset_y)
            self.server_train_dataloader = list(DataLoader(self.server_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False))

            self.server_test_dataset_x = torch.stack(self.server_test_dataset_x).permute(1, 0, 2, 3)
            self.server_test_dataset_y = torch.stack(self.server_test_dataset_y).permute(1, 0)
            self.server_test_dataset = ServerDataset(self.server_test_dataset_x, self.server_test_dataset_y)
            self.server_test_dataloader = list(DataLoader(self.server_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False))
        elif 'cifar10' in self.dataset:
            #print(torch.stack(self.server_train_dataset_x).shape) #torch.Size([100, 476, 32, 32, 3])
            self.server_train_dataset_x = torch.stack(self.server_train_dataset_x).permute(1, 0, 2, 3, 4)# ([100, 476, 32, 32, 3]) N,B,W,H,C->B,N,W,H,C
            self.server_train_dataset_y = torch.stack(self.server_train_dataset_y).permute(1, 0)#B,N [100, 476]
            self.server_train_dataset = ServerCIFAR10Dataset(self.server_train_dataset_x, self.server_train_dataset_y)
            self.server_train_dataloader = list(DataLoader(self.server_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))

            self.server_test_dataset_x = torch.stack(self.server_test_dataset_x).permute(1, 0, 2, 3, 4)
            self.server_test_dataset_y = torch.stack(self.server_test_dataset_y).permute(1, 0)
            self.server_test_dataset = ServerCIFAR10Dataset(self.server_test_dataset_x, self.server_test_dataset_y)
            self.server_test_dataloader = list(DataLoader(self.server_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True))   
    
    def get_user_embeddings(self, args, users):
        base = args.base
        user_feats = [] # [N,B,F]
        for i, user in enumerate(users): 
            if base == 'feature':
                feat = user.features # [Len, F] #torch.Size([520, 10])
                #print(feat.shape) [F]
            elif base == 'model':
                feat = sd_matrixing(user.model.state_dict())
            elif base == 'feature_label':
                feat = user.representations           
            elif base == 'feature_label_model':
                feat = sd_matrixing(user.model.state_dict())
            user_feats.append(feat)

        user_feats = torch.stack(user_feats)
        #print(user_feats.shape)# #torch.Size([100, 1280])
        return user_feats
    
    def get_random_mask(self, features, r):
        probs = torch.full(features.shape, 1 / r)
        mask = torch.bernoulli(probs)
        #print(mask)
        return mask.to(self.device)

    def get_loss_masked_features(self, model, features, mask, ogb, noise, loss_t):
        if ogb:
            if noise == 'mask':
                masked_features = features * (1 - mask)
            elif noise == "normal":
                noise = torch.normal(0.0, 1.0, size=features.shape).to(self.device)
                masked_features = features + (noise * mask)
            #print(features)
            #print(masked_features)
            logits, Adj = model(features, masked_features)
            indices = mask > 0
            #print(indices)

            if loss_t == 'bce':
                features_sign = torch.sign(features).cuda() * 0.5 + 0.5
                loss = F.binary_cross_entropy_with_logits(logits[indices], features_sign[indices], reduction='mean')
            elif loss_t == 'mse':
                loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
        else:
            masked_features = features * (1 - mask)
            logits, Adj = model(features, masked_features)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        return loss, Adj
    
    def learn_client_graph(self, args, user_embeddings, graph_include_self=False, init_adj=None):
        # calculate learn_adj_loss
        ogb = True
        for epoch in range(1, args.epochs_adj + 1):
            self.graph_model_SLAPS.train()
            #print(user_embeddings)
            #print(user_embeddings.shape) #torch.Size([100, 64])
            mask = self.get_random_mask(user_embeddings, args.ratio)
            loss_adj, raw_adj = self.get_loss_masked_features(self.graph_model_SLAPS, user_embeddings, mask, ogb, args.noise, args.loss)
            self.optimizer_graph_model_SLAPS.zero_grad()
            loss_adj.backward()
            self.optimizer_graph_model_SLAPS.step()

        self.save_adj(args, raw_adj)
        return raw_adj

    def save_adj(self, args, adj):
        if args.sparse:
            adj = dgl_graph_to_torch_sparse(adj)
            #u, v = adj.edges()
            #u = u.detach().cpu().numpy()
            #v = v.detach().cpu().numpy()
            #size = u.shape[0]
            #data = np.ones_like(u)
            #adj = sp.coo_matrix((data, (u, v)), shape=(size, size)).toarray()
            print(adj)
            #adj = adj.detach().cpu().toarray()
        else:
            adj = adj.detach().cpu().numpy()
        self.metrics['adj'].append(adj)

        
    def learn_adj(self, args, graph_include_self=False):
        
        user_embeddings = self.get_user_embeddings(args, users=self.selected_users).to(self.device)
        adj = self.learn_client_graph(args, user_embeddings = user_embeddings, init_adj = self.original_A)#[N,N]
        print(adj.shape)
        init_adj = torch.eye(adj.shape[0]).to(self.device)
        #graph_skip_conn = args.tau
        
        #if graph_skip_conn not in (0, None):
            #adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

        #if args.sparse:
            #init_adj_torch_sparse = copy.deepcopy(init_adj)
            #init_adj = torch_sparse_to_dgl_graph(init_adj)

        # Structure Bootstrapping
        
        print("raw_learned_adj", adj)
        if (1 - args.tau): 
        #and (args.c == 0 or epoch % args.c == 0):
            if args.sparse:
                learned_adj_torch_sparse = dgl_graph_to_torch_sparse(adj)
                adj = init_adj.cpu() * args.tau + learned_adj_torch_sparse * (1 - args.tau)
                adj = torch_sparse_to_dgl_graph(adj).to(self.device)
            else:
                print(args.tau)
                adj = init_adj * args.tau + adj.detach() * (1 - args.tau)

        print("after_strap", adj)
        # generate graph-based user feature
        # define GCN/GAT model
        self.learned_adj = adj
        
    def train_server_gcn_with_agg_clients_shakespeare(self, args):
        self.learn_adj(args)
        batch_size = self.batch_size

        server_train_dataloader = self.server_train_dataloader
       
        self.model.train()
        self.gcn.train()
        for epoch_i in range(args.server_epoch):
            adj_loss = 0

            updated_graph_encoding = []
            if epoch_i == self.server_epoch - 1:
                server_train_dataloader = self.server_train_dataloader

            for x, y, _ in server_train_dataloader: 
            
                # x [B,N,seq_len]
                # y [B,N,seq_len]
                
                if x.shape[0] != batch_size:
                        x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2])
                        y_buff = torch.zeros(batch_size, y.shape[1])
                        x_buff[: x.shape[0], :, :] = x
                        x_buff[x.shape[0] :, :, :] = x[-1].repeat(
                            batch_size - x.shape[0], 1, 1
                        )
                        y_buff[: x.shape[0], :, :] = y
                        y_buff[x.shape[0] :, :, :] = y[-1].repeat(
                            batch_size - x.shape[0], 1, 1
                        )
                        x = x_buff
                        y = y_buff


                batch_num, node_num = x.shape[0], x.shape[1] # [B,N,seq_len]
                x = x.flatten(0, 1).to(self.device) #
                y = y.flatten(0, 1).to(self.device)

                client_encode = self.model.forward_encoder(x) # B*N, seq_len, hidden_size

                #if args.sparse:
                    #graph_encoding = client_encode.view(node_num, batch_num, client_encode.shape[1]) # N, B, H 
                    #graph_encoding = self.gcn(self.learn_g, graph_encoding)
                    #graph_encoding_to_model = graph_encoding.permute(1,0,2).flatten(0,1)
                #else:
                client_encode_to_graph_learner = client_encode.view(batch_num, node_num, client_encode.shape[1], client_encode.shape[2]) #  B, N, seq_len, hidden_size
                graph_encoding = self.gcn(client_encode_to_graph_learner, self.learned_adj)  #  B, N, seq_len, hidden_size  # N x B x D*num_layers x F 
                graph_encoding_to_model = graph_encoding.flatten(0,1) # B*N, seq_len, hidden_size  

                if epoch_i == self.server_epoch - 1: # B*N, seq_len, hidden_size
                    updated_graph_encoding.append(graph_encoding.permute(1,0,2,3).detach().clone().cpu()) #  N, B, seq_len, hidden_size
                else:
                    output=self.model.forward_decoder(graph_encoding_to_model) #torch.Size([3200, 10])
                    #adj_loss += loss
                    loss_vec = self.shakespeare_loss(output, y.long())

                #if weights is not None:
                    #weights = weights.to(self.device)
                    #loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
                #else:
                    loss = loss_vec.mean()
                    self.server_optimizer_gcn.zero_grad()
                    loss.backward()
                    self.server_optimizer_gcn.step()
            
            #self.optimizer_graph_learner.zero_grad()
            #adj_loss.backward()
            #self.optimizer_graph_learner.step()
            #self.server_optimizer_gcn.step()

        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  #N, L, seq_len, hidden_size
        #print(updated_graph_encoding.shape)
        self.send_server_graph_encoding(args, updated_graph_encoding, train=True)

    def _eval_server_gcn_with_agg_clients_shakespeare(self, args):
        #self.learn_adj(args)
        batch_size = self.batch_size

        server_train_dataloader = self.server_train_dataloader
       
        self.model.train()
        self.gcn.train()
        for epoch_i in range(args.server_epoch):
            updated_graph_encoding = []
            if epoch_i == self.server_epoch - 1:
                server_train_dataloader = self.server_train_dataloader

            for x, y, _ in server_train_dataloader: 
                if x.shape[0] != batch_size:
                        x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
                        y_buff = torch.zeros(batch_size, y.shape[1])
                        x_buff[: x.shape[0], :, :, :] = x
                        x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                            batch_size - x.shape[0], 1, 1, 1
                        )
                        y_buff[: x.shape[0], :] = y
                        y_buff[x.shape[0] :, :] = y[-1].repeat(
                            batch_size - x.shape[0], 1
                        )
                        x = x_buff
                        y = y_buff


                batch_num, node_num = x.shape[0], x.shape[1]
                x = x.flatten(0, 1).to(self.device) #[B*N,W,H]  #[B*N,W,H,C]
                y = y.flatten(0, 1).to(self.device)

                client_encode = self.model.forward_encoder(x) # B, H

                #if args.sparse:
                    #graph_encoding = client_encode.view(node_num, batch_num, client_encode.shape[1]) # N, B, H 
                    #graph_encoding = self.gcn(self.learn_g, graph_encoding)
                    #graph_encoding_to_model = graph_encoding.permute(1,0,2).flatten(0,1)
                #else:
                graph_encoding = client_encode.view(batch_num, node_num, client_encode.shape[1], client_encode.shape[2]) #  B, N, H 
                #graph_encoding = self.gcn(client_encode_to_graph_learner, self.learned_adj)  #[B,N,F]   
                updated_graph_encoding.append(graph_encoding.permute(1,0,2,3).detach().clone().cpu()) #  N, B, seq_len, hidden_size

            

        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N, L, H  torch.Size([100, 448, 1280])
        #print(updated_graph_encoding.shape)
        self.send_server_graph_encoding(args, updated_graph_encoding, train=False)
        
    def train_server_gcn_with_agg_clients_normal(self, args):
        self.learn_adj(args)
        batch_size = self.batch_size

        server_train_dataloader = self.server_train_dataloader
       
        self.model.train()
        self.gcn.train()
        for epoch_i in range(args.server_epoch):
            adj_loss = 0
            
            updated_graph_encoding = []
            if epoch_i == self.server_epoch - 1:
                server_train_dataloader = self.server_train_dataloader

            for x, y, _ in server_train_dataloader: 
                
                #x, y, = x.to(self.device), y.to(self.device) #476 #[B,N,W,H]  #[B,N,W,H,C]
                #print(x.shape) #torch.Size([32, 28, 100, 28])
                #print(x.type()) torch.cuda.FloatTensor
                #print(y.type()) torch.cuda.LongTensor
                # mnist [N,B,W,H]  cifar10 [N,B,W,H,C]
                # [N,B]
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                    if x.shape[0] != batch_size:
                        x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3], x.shape[4]) 
                        y_buff = torch.zeros(batch_size, y.shape[1])
                        x_buff[: x.shape[0], :, :, :] = x
                        x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                            batch_size - x.shape[0], 1
                        )
                        y_buff[: x.shape[0], :] = y
                        y_buff[x.shape[0] :, :] = y[-1].repeat(
                            batch_size - x.shape[0], 1
                        )
                        x = x_buff
                        y = y_buff
                else:
                    if x.shape[0] != batch_size:
                        x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
                        y_buff = torch.zeros(batch_size, y.shape[1])
                        x_buff[: x.shape[0], :, :, :] = x
                        x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                            batch_size - x.shape[0], 1, 1, 1
                        )
                        y_buff[: x.shape[0], :] = y
                        y_buff[x.shape[0] :, :] = y[-1].repeat(
                            batch_size - x.shape[0], 1
                        )
                        x = x_buff
                        y = y_buff


                batch_num, node_num = x.shape[0], x.shape[1]
                x = x.flatten(0, 1).to(self.device) #[B*N,W,H]  #[B*N,W,H,C]
                y = y.flatten(0, 1).to(self.device)

                client_encode = self.model.features(x) # B, H 
                if 'cifar10' in self.dataset:
                    client_encode = self.model.features_flatten(client_encode) # B*N, H

                #if args.sparse:
                    #graph_encoding = client_encode.view(node_num, batch_num, client_encode.shape[1]) # N, B, H 
                    #graph_encoding = self.gcn(self.learn_g, graph_encoding)
                    #graph_encoding_to_model = graph_encoding.permute(1,0,2).flatten(0,1)
                #else:
                client_encode_to_graph_learner = client_encode.view(batch_num, node_num, client_encode.shape[1]) #  B, N, H 
                graph_encoding = self.gcn(client_encode_to_graph_learner, self.learned_adj)  #[B,N,F]   
                graph_encoding_to_model = graph_encoding.flatten(0,1) #[B*N,F]   

                if epoch_i == self.server_epoch - 1:
                    updated_graph_encoding.append(graph_encoding.permute(1,0,2).detach().clone().cpu()) # N, B, H  #torch.Size([100, 32, 1280])
                else:
                    output=self.model.linears(client_encode, graph=True, graph_encoding = graph_encoding_to_model) #torch.Size([3200, 10])
                    loss=self.ce_loss(output, y.long())
                    #adj_loss += loss
                    self.server_optimizer_gcn.zero_grad()
                    loss.backward()
                    self.server_optimizer_gcn.step()
            
            #self.optimizer_graph_learner.zero_grad()
            #adj_loss.backward()
            #self.optimizer_graph_learner.step()
            #self.server_optimizer_gcn.step()

        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N, L, H  torch.Size([100, 448, 1280])
        #print(updated_graph_encoding.shape)
        self.send_server_graph_encoding(args, updated_graph_encoding, train=True)
                
    def _eval_server_gcn_with_agg_clients_normal(self, args):
        for user in self.selected_users:
            user.get_test_data_feature()
        self.learn_adj(args)
        #server_train_dataloader = DataLoader(self.server_datasets['train'], batch_size=self.hparams.server_batch_size, shuffle=True)
        batch_size = self.batch_size
        server_test_dataloader = self.server_test_dataloader
       
        self.model.train()
        self.gcn.train()
        updated_graph_encoding = []
  
        for x, y,  _ in server_test_dataloader: 
            if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                    if x.shape[0] != batch_size:
                        x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3], x.shape[4]) 
                        y_buff = torch.zeros(batch_size, y.shape[1])
                        x_buff[: x.shape[0], :, :, :] = x
                        x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                            batch_size - x.shape[0], 1
                        )
                        y_buff[: x.shape[0], :] = y
                        y_buff[x.shape[0] :, :] = y[-1].repeat(
                            batch_size - x.shape[0], 1
                        )
                        x = x_buff
                        y = y_buff
            else:
                    if x.shape[0] != batch_size:
                        x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
                        y_buff = torch.zeros(batch_size, y.shape[1])
                        x_buff[: x.shape[0], :, :, :] = x
                        x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                            batch_size - x.shape[0], 1, 1, 1
                        )
                        y_buff[: x.shape[0], :] = y
                        y_buff[x.shape[0] :, :] = y[-1].repeat(
                            batch_size - x.shape[0], 1
                        )
                        x = x_buff
                        y = y_buff
            # [N,B,W,H]
            # [N,B]
            batch_num, node_num = x.shape[0], x.shape[1]
            x = x.flatten(0, 1).to(self.device) #[B*N,W,H]
            y = y.flatten(0, 1).to(self.device)
    
            client_encode = self.model.features(x) # B, H 
            if 'cifar10' in self.dataset:
                client_encode = self.model.features_flatten(client_encode) # B*N, H
            
            client_encode_to_graph_learner = client_encode.view(batch_num, node_num, client_encode.shape[1]) #  B, N, H 
            graph_encoding = self.gcn(client_encode_to_graph_learner, self.learned_adj)  #[B,N,F]   
        
            updated_graph_encoding.append(graph_encoding.permute(1,0,2).detach().clone().cpu()) # N, B, H  #torch.Size([100, 32, 1280])
                
        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  
        self.send_server_graph_encoding(args, updated_graph_encoding)

    def train_server_gcn_with_agg_clients_traffic(self, args):
        #server_train_dataloader = DataLoader(self.server_datasets['train'], batch_size=self.hparams.server_batch_size, shuffle=True)
        server_train_dataloader = self.server_train_dataloader
       
        batch_size = self.batch_size
        self.model.train()
        self.gcn.train()
        for epoch_i in range(args.server_epoch):
            updated_graph_encoding = []
            if epoch_i == self.server_epoch - 1:
                server_train_dataloader = self.server_train_dataloader
            
            for x, y, graph_encoding, _ in server_train_dataloader: 
                # B x T x N x F
                
                if x.shape[0] != batch_size:
                        x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])  # T x (B x N) x F
                        y_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
                        x_buff[: x.shape[0], :, :, :] = x
                        x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                            batch_size - x.shape[0], 1, 1, 1
                        )
                        y_buff[: x.shape[0], :, :, :] = y
                        y_buff[x.shape[0] :, :, :, :] = y[-1].repeat(
                            batch_size - x.shape[0], 1, 1, 1
                        )
                        x = x_buff
                        y = y_buff

                batch_num, node_num = x.shape[0], x.shape[2]
                x = x.permute(1, 0, 2, 3)
                y = y.permute(1, 0, 2, 3)
                x_ = x.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)  # T x (B x N) x F #torch.Size([12, 64, 1, 2])
                y_ = y.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)

                h_encode = self.model.forward_encoder(x_) #  D*num_layers x (B x N) x F  #torch.Size([2, 64, 64])
                #print(h_encode.shape)
                graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # N x B x D*num_layers x F 
                # D*num_layers x B x N x F 
                graph_encoding = self.gcn(graph_encoding, self.learned_adj) # N x B x D*num_layers x F 
                graph_encoding_to_model = graph_encoding.permute(2, 1, 0, 3).flatten(1, 2).float().to(self.device) # D*num_layers x (B x N) x F
                #print(graph_encoding.shape)
                if epoch_i == self.server_epoch - 1: #这里的意思是说，轮到最后一个epoch的时候，每个batch的graph_encoding都会加进去（这是第二层for循环啊）  两层for循环啊，
                    updated_graph_encoding.append(graph_encoding.detach().clone().cpu()) # N x B x D*num_layers x F 
                else:
                    y_pred = self.model.forward_decoder(y_, h_encode , return_encoding=False, server_graph_encoding=graph_encoding_to_model)
                    loss = nn.MSELoss()(y_pred, y_)
                    self.server_optimizer_gcn.zero_grad()
                    loss.backward()
                    self.server_optimizer_gcn.step()
                    #global_step += 1

        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N x B x D*num_layers x F 
        #updated_graph_encoding = updated_graph_encoding.permute(2, 1, 0, 3).flatten(1, 2) # L x (B x N) x F
        self.send_server_graph_encoding(args, updated_graph_encoding, train=True)

    def _eval_server_gcn_with_agg_clients_traffic(self, args):
        server_test_dataloader = self.server_test_dataloader
        batch_size = self.batch_size

        updated_graph_encoding = []
       
        for x, y, graph_encoding, _ in server_test_dataloader: 
                # B x T x N x F
                #print(x.shape) #torch.Size([64, 12, 207, 2])
                if x.shape[0] != batch_size:
                        x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3]) 
                        y_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
                        x_buff[: x.shape[0], :, :, :] = x
                        x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                            batch_size - x.shape[0], 1, 1, 1
                        )
                        y_buff[: x.shape[0], :, :, :] = y
                        y_buff[x.shape[0] :, :, :, :] = y[-1].repeat(
                            batch_size - x.shape[0], 1, 1, 1
                        )
                        x = x_buff
                        y = y_buff

                batch_num, node_num = x.shape[0], x.shape[2]
                x = x.permute(1, 0, 2, 3)
                y = y.permute(1, 0, 2, 3)
                x_ = x.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)  # T x (B x N) x F #torch.Size([12, 64, 1, 2])
                y_ = y.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)

                h_encode = self.model.forward_encoder(x_) # T x (B x N) x F 
                #print("eval_h_encode", h_encode.shape) #([2, 13248, 64])
                graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # N x B x T x F 
                #print("graph_encoding",graph_encoding.shape) #([207, 64, 2, 64])
                #graph_encoding = self.gcn(self.g, graph_encoding) # # N x B x T x F 
                #print("gcn_graph_encoding",graph_encoding.shape) #([207, 64, 2, 64])
                graph_encoding_to_model = graph_encoding.permute(2, 1, 0, 3).flatten(1, 2).float().to(self.device) # T x (B x N) x F
                #print(graph_encoding.shape)
                updated_graph_encoding.append(graph_encoding.detach().clone().cpu()) # B x T x N x F

        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N x B x L x F    
        #updated_graph_encoding = updated_graph_encoding.permute(2, 1, 0, 3).flatten(1, 2) # L x (B x N) x F
        self.send_server_graph_encoding(args, updated_graph_encoding)
    
    def train_server_gcn_with_agg_clients(self, args):
        if 'metr-la' in self.dataset:
            self.train_server_gcn_with_agg_clients_traffic(args)
        elif 'shakespeare' in self.dataset:
            self.train_server_gcn_with_agg_clients_shakespeare(args)
        else:
            self.train_server_gcn_with_agg_clients_normal(args)
        
    def _eval_server_gcn_with_agg_clients(self, args):
        if 'metr-la' in self.dataset:
            self._eval_server_gcn_with_agg_clients_traffic(args)
        elif 'shakespeare' in self.dataset:
            self._eval_server_gcn_with_agg_clients_shakespeare(args)
        else:
            self._eval_server_gcn_with_agg_clients_normal(args)
        
    def send_server_graph_encoding(self, args, updated_graph_encoding, train=False):
        #print("updated_graph_encoding.shape", updated_graph_encoding.shape) #([207, 24000, 2, 64])
        for i, user in enumerate(self.selected_users):
            # N x B x D*num_layers x F 
            if 'metr-la' in self.dataset:
                updated_graph_encoding_to_user = updated_graph_encoding[i:i+1, :, :, :].permute(1, 2, 0, 3) # B x D*num_layers x N x  F
            elif 'shakespeare' in self.dataset:
                #N, L, seq_len, hidden_size
                updated_graph_encoding_to_user = updated_graph_encoding[i:i+1, :, :, :].permute(1,0,2,3) # N, L, seq_len, hidden_size  #torch.Size([1, 10784, 80, 256]) -> # L, 1, seq_len, hidden_size
            else:
                updated_graph_encoding_to_user = updated_graph_encoding[i:i+1].permute(1,0,2) # L, 1, H  
            user.set_server_graph_encoding(args, updated_graph_encoding = updated_graph_encoding_to_user, train = train)  
            #print(updated_graph_encoding.shape) #torch.Size([24000, 2, 1, 64])

    def train(self, args):
       
        train_start= time.time() 
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
        
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.send_parameters(self.users, mode=self.mode)
            #self.send_parameters_SFL(self.users)
            
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
            
           
            self.train_server_gcn_with_agg_clients(args)
            self._eval_server_gcn_with_agg_clients(args)
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

    