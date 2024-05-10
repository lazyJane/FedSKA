from fcntl import F_SETLEASE
import torch
from FLAlgorithms.users.userbase import User
import copy
from utils.model_utils import *
from collections import defaultdict
from utils.traffic_utils import *
from datasets.st_datasets import *

class userSKA(User):
    def __init__(self,  args, id, model, train_dataset, test_dataset, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.global_param = self.model.state_dict()
        self.server_param = self.model.state_dict()
        self.batch_adj = args.batch_adj
        self.graph_encode_len = 0

        #self.server_graph_encoding = torch.zeros(self.batch_size, args.gru_num_layers, 1, args.user_feature_hidden_size) # default_server_graph_encoding 
        # B x T x N x F
        self.representations = []
        self.labels = []
        self.features = []
        self.mode = "Train"

    def set_server_graph_encoding(self, args, updated_graph_encoding, train=False):
        drop_last = True
        if train:
            #print("self.train_dataset.graph_encoding", self.train_dataset.graph_encoding.shape) #([23974, 2, 1, 64])
            #print("self.train_dataset.x.shape", self.train_dataset.x.shape) #([23974, 12, 1, 2])
            #print("updated_graph_encoding.shape", updated_graph_encoding.shape) #([24000, 2, 1, 64])
            #print(self.train_dataset.data.shape[0])#476
            #print(updated_graph_encoding.shape)#torch.Size([448, 1, 1280])
            if 'mnist' in self.dataset:
                updated_graph_encoding = updated_graph_encoding[:self.train_dataset.data.shape[0]]
            elif 'cifar10' in self.dataset:
                lack = self.train_dataset.data.shape[0] - updated_graph_encoding.shape[0]
                fill_updated_graph_encoding = updated_graph_encoding[0:lack]
                updated_graph_encoding = torch.cat((updated_graph_encoding, fill_updated_graph_encoding), dim=0)
            #print("updated_train_graph_encoding.shape", updated_graph_encoding.shape)
            elif 'shakespeare' in self.dataset:
               #print(updated_graph_encoding.shape) # L, 1, seq_len, hidden_size
               #print(self.train_dataset.data.shape[0])
               if updated_graph_encoding.shape[0] >=  self.train_dataset.data.shape[0]:
                    updated_graph_encoding = updated_graph_encoding[:self.train_dataset.data.shape[0]]
               else:
                    graph_encode_len = updated_graph_encoding.shape[0] 
                    lack = self.train_dataset.data.shape[0] - updated_graph_encoding.shape[0]
                    while lack > graph_encode_len:
                        fill_updated_graph_encoding = updated_graph_encoding
                        updated_graph_encoding = torch.cat((updated_graph_encoding, fill_updated_graph_encoding), dim=0)
                        lack = self.train_dataset.data.shape[0] - updated_graph_encoding.shape[0]
                    fill_updated_graph_encoding = updated_graph_encoding[0:lack]
                    updated_graph_encoding = torch.cat((updated_graph_encoding, fill_updated_graph_encoding), dim=0)
                
            self.train_dataset.graph_encoding = updated_graph_encoding
           
            self.trainloader = list(DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=train, drop_last=drop_last))
        else:
            if 'mnist' in self.dataset:
                updated_graph_encoding = updated_graph_encoding[:self.test_dataset.data.shape[0]]# 1, B, H 
            elif 'cifar10' in self.dataset:
                lack = self.test_dataset.data.shape[0] - updated_graph_encoding.shape[0]
                fill_updated_graph_encoding = updated_graph_encoding[0:lack]
                updated_graph_encoding = torch.cat((updated_graph_encoding, fill_updated_graph_encoding), dim=0)
            elif 'shakespeare' in self.dataset:
                if updated_graph_encoding.shape[0] >=  self.test_dataset.data.shape[0]:
                    updated_graph_encoding = updated_graph_encoding[:self.test_dataset.data.shape[0]]
                else:
                    graph_encode_len = updated_graph_encoding.shape[0] 
                    lack = self.train_dataset.data.shape[0] - updated_graph_encoding.shape[0]
                    while lack > graph_encode_len:
                        fill_updated_graph_encoding = updated_graph_encoding
                        updated_graph_encoding = torch.cat((updated_graph_encoding, fill_updated_graph_encoding), dim=0)
                        lack = self.train_dataset.data.shape[0] - updated_graph_encoding.shape[0]
                    fill_updated_graph_encoding = updated_graph_encoding[0:lack]
                    updated_graph_encoding = torch.cat((updated_graph_encoding, fill_updated_graph_encoding), dim=0)
            #print("updated_test_graph_encoding.shape", updated_graph_encoding.shape)
            self.test_dataset.graph_encoding = updated_graph_encoding
            self.testloader = list(DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=train, drop_last=drop_last))

    def fit_epochs(self, glob_iter = 0, lr_decay = True, unseen = False, return_encode = False, return_grad = False):
        if 'shakespeare' in self.dataset: 
            self.fit_epochs_shakespeare(glob_iter, lr_decay)
            return 
        if 'metr' in self.dataset:
            self.fit_epochs_metr_la(glob_iter, lr_decay)
            return 
        
        batch_size = self.batch_size
        features = []
        labels = []
        n_steps = 0
        self.model.train()
        for step in range(self.E):
            for x, y, graph_encoding, _ in self.trainloader:
                    #if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                        # B,W,H,C
                    if x.shape[0] != batch_size:
                            x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3]) 
                            y_buff = torch.zeros(batch_size)
                            graph_encoding_buff = torch.zeros(batch_size, graph_encoding.shape[1], graph_encoding.shape[2])
                            x_buff[: x.shape[0], :, :, :] = x
                            x_buff[x.shape[0] :, :, :, :]  = x[-1].repeat(
                                batch_size - x.shape[0], 1, 1, 1
                            )
                            y_buff[: x.shape[0]] = y
                            y_buff[x.shape[0] :] = y[-1].repeat(
                                batch_size - x.shape[0]
                            )
                            graph_encoding_buff[: x.shape[0]] = graph_encoding
                            graph_encoding_buff[x.shape[0] :, :, :] = graph_encoding[-1].repeat(
                                 batch_size - x.shape[0], 1, 1
                            )
                            x = x_buff
                            y = y_buff.to(torch.int64)
                            graph_encoding = graph_encoding_buff
                    
                    #else:
                        # B,W,H
                    '''
                        print(x.shape)
                        if x.shape[0] != batch_size:
                            x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2])
                            y_buff = torch.zeros(batch_size)
                            x_buff[: x.shape[0], :, :] = x
                            x_buff[x.shape[0] :, :, :] = x[-1].repeat(
                                batch_size - x.shape[0], 1, 1
                            )
                            y_buff[: x.shape[0], :] = y
                            y_buff[x.shape[0] :] = y[-1].repeat(
                                batch_size - x.shape[0]
                            )
                            x = x_buff
                            y = y_buff
                    '''
                    n_steps += 1
                    x, y, graph_encoding = x.to(self.device), y.to(self.device), graph_encoding.to(self.device)
                    graph_encoding = graph_encoding.flatten(0, 1) # (B x N) x F
                    
                    self.optimizer.zero_grad()
                    if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                        feature_embedding = self.model.features(x)
                        feature_embedding = self.model.features_flatten(feature_embedding) 
                        output=self.model(x)
                        loss=self.ce_loss(output, y)
                    else:
                        feature_embedding = self.model.features(x)
                        #print(feature_embedding.shape)
                        output=self.model(x)['output']
                        #print(output.shape)
                        loss=self.loss(output, y)
                   
                    #print(F.cosine_similarity(graph_encoding,feature_embedding).shape)#torch.Size([32])
                    loss_feature = 0
                    loss_feature += F.cosine_similarity(graph_encoding,feature_embedding).mean()
                    #loss_feature += F.pairwise_distance(graph_encoding,feature_embedding, p=2).mean()
                    loss = loss - 0.01*loss_feature
                    loss.backward()
                    self.optimizer.step()

                    if step == self.E - 1: #这里的意思是说，轮到最后一个epoch的时候，每个batch的graph_encoding都会加进去（这是第二层for循环啊）  两层for循环啊，
                        features.append(feature_embedding.detach().clone().cpu()) # B x F
                        batch_size = x.shape[0]
                        label_one_hot = torch.zeros(batch_size, self.n_classes) # B x N_CLASSES
                        for i, index  in zip(range(batch_size), y): 
                            label_one_hot[i][int(index)-1] = 1
                        labels.append(label_one_hot.clone().cpu()) #
            if lr_decay:
                if unseen:
                    self.lr_scheduler.step(step)
                else:
                    self.lr_scheduler.step(glob_iter) #存疑
        if self.batch_adj:
            self.features = torch.stack(features) # [N_batch, B, F]
            self.labels = torch.stack(labels) # [N_batch, B, C]
            self.representations = torch.cat([self.features, self.labels],dim=-1)
        else:
            self.features = torch.cat(features, dim=0) # [Len, F]
            self.features = self.features.mean(dim=0)
            #print(self.features)
            #input()
            #print(self.features.shape)
            #input()
            self.labels = torch.cat(labels, dim=0)
            self.labels = self.labels.mean(dim=0)
            self.representations = torch.cat([self.features, self.labels])
                
        return n_steps
    
    def get_test_data_feature(self):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for x, y, graph_encoding, _ in self.testloader:
                x, y, graph_encoding = x.to(self.device), y.to(self.device), graph_encoding.to(self.device)
                graph_encoding = graph_encoding.flatten(0, 1) # (B x N) x F
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:       
                    feature_embedding = self.model.features(x)
                    feature_embedding = self.model.features_flatten(feature_embedding) 
                else: 
                    feature_embedding = self.model.features(x)

                features.append(feature_embedding.detach().clone().cpu()) # B x F
                batch_size = x.shape[0]
                label_one_hot = torch.zeros(batch_size, self.n_classes) # B x N_CLASSES
                for i, index  in zip(range(batch_size), y): 
                    label_one_hot[i][int(index)-1] = 1
                labels.append(label_one_hot.clone().cpu()) #

        self.features = torch.cat(features, dim=0) # [Len, F]
        self.features = self.features.mean(dim=0)
        self.labels = torch.cat(labels, dim=0)
        self.labels = self.labels.mean(dim=0)
        self.representations = torch.cat([self.features, self.labels])
    
    def fit_batches(self, glob_iter = 0, count_labels=False, lr_decay=True, unseen = False, return_encode = False, return_grad = False):
        self.model.train()
        num_batches = 6
        record_start = self.local_epochs//num_batches - 1
        #print(record_start)
        features = []
        for epoch in range(1, self.local_epochs + 1):
            result =self.get_next_train_batch(count_labels=count_labels)
            X, y, graph_encoding = result['X'], result['y'], result['graph_encoding']#[B,W,H,C]
            X, y, graph_encoding = X.to(self.device), y.to(self.device), graph_encoding.to(self.device)
            graph_encoding = graph_encoding.flatten(0, 1) # (B x N) x F
            #if count_labels:
                #self.update_label_counts(result['labels'], result['counts'])

            self.optimizer.zero_grad()
            if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                feature_embedding = self.model.features(X)
                feature_embedding = self.model.features_flatten(feature_embedding) #[B,F]
                output=self.model(X)
                loss=self.ce_loss(output, y)
            else:
                output=self.model(X)['output']
                loss=self.loss(output, y)
                
            #print(F.cosine_similarity(graph_encoding,feature_embedding).shape)#torch.Size([32])
            loss_feature = 0
            loss_feature += F.cosine_similarity(graph_encoding,feature_embedding).mean()#[B,F]
            #loss_feature += F.pairwise_distance(graph_encoding,feature_embedding, p=2).mean()
            #loss = 0.5*loss + 0.5*loss_feature
            loss = 0.5*(loss - loss_feature)
            #loss = 0.8*loss - loss_feature
           
            loss.backward()
            self.optimizer.step()#self.plot_Celeb)

            if epoch in range(record_start*num_batches, record_start*num_batches+num_batches): #这里的意思是说，轮到最后一个epoch的时候，每个batch的graph_encoding都会加进去（这是第二层for循环啊）  两层for循环啊，
                #print(feature_embedding)
                features.append(feature_embedding.detach().clone().cpu()) # B x F
        
        if lr_decay:
            if unseen:
                self.lr_scheduler.step(epoch)
            else:
                self.lr_scheduler.step(glob_iter) #存疑
        self.features = torch.cat(features, dim=0) # [Len, F]
        self.features = self.features.mean(dim=0)
        
    def fit_epochs_shakespeare(self, glob_iter, lr_decay):
        batch_size = self.batch_size
        self.model.train()
        features = []
        for step in range(self.E):
            graph_cur_len = 0
            for x, y, graph_encoding, indices in self.trainloader:
                # x.shape [32,80]
                graph_cur_len += batch_size
                if x.shape[0] != batch_size:
                    x_buff = torch.zeros(batch_size, x.shape[1]) 
                    y_buff = torch.zeros(batch_size, x.shape[1])
                    x_buff[: x.shape[0], :] = x
                    x_buff[x.shape[0] :, :] = x[-1].repeat(
                        batch_size - x.shape[0], 1
                    )
                    y_buff[: x.shape[0], :] = y
                    y_buff[x.shape[0] :, :] = y[-1].repeat(
                        batch_size - x.shape[0], 1
                    )
                    x = x_buff.long()
                    y = y_buff.long()
                x = x.to(self.device)
                y = y.to(self.device)
                graph_encoding = graph_encoding.flatten(0,1).to(self.device)

                # n_samples += y.size(0)
                chunk_len = y.size(1)
                self.optimizer.zero_grad()
                
                h_encode = self.model.forward_encoder(x) #[32,80,256]
                y_pred = self.model(x)
                loss_vec = self.shakespeare_loss(y_pred, y)

                #if weights is not None:
                    #weights = weights.to(self.device)
                    #loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
                #else:
                loss = loss_vec.mean()
                loss_feature = 0
                #print(graph_encoding.shape)
                #print(h_encode.shape)
                #print(F.cosine_similarity(graph_encoding,h_encode,dim=-1).shape) #[32,80]
                if graph_cur_len <= self.graph_encode_len:
                    loss_feature += F.cosine_similarity(graph_encoding,h_encode,dim=-1).mean() #[32,80]
                loss -= 0.1* loss_feature
                loss.backward()
                self.optimizer.step()

                
                if step == self.E - 1: #这里的意思是说，轮到最后一个epoch的时候，每个batch的graph_encoding都会加进去（这是第二层for循环啊）  两层for循环啊，
                    features.append(h_encode.detach().clone().cpu()) #D*num_layers x (B x N) x F
     

        if self.batch_adj:
            self.features = torch.stack(features) # [N_batch, B, F]
            #self.labels = torch.stack(labels) # [N_batch, B, C]
            #self.representations = torch.cat([self.features, self.labels],dim=-1)
        else:
            self.features = torch.cat(features, dim=0) #[32,80,256]
            self.features = self.features.mean(dim=0) #D*num_layers x F
            self.features = self.features.mean(dim=1) # F
            #self.labels = torch.cat(labels, dim=0)
            #self.labels = self.labels.mean(dim=0)
            #self.representations = torch.cat([self.features, self.labels])

    def fit_epochs_metr_la(self, glob_iter, lr_decay):
        total_loss = []
        batch_size = self.batch_size
        self.model.train()
        features = []
        labels = []
        for step in range(self.E):
            for x, y, graph_encoding, _ in self.trainloader: 
                #print(x.shape)# [64,12,1,2]  # B x T x N x F
                #print(y.shape)
                #input()
                self.optimizer.zero_grad()
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
                
                # B x D*num_layers x N x  F
                graph_encoding = graph_encoding.permute(1, 0, 2, 3).flatten(1, 2).float().to(self.device) #  D*num_layers x (B x N) x F
                # L x (B x N) x F
                x = x.permute(1, 0, 2, 3)
                y = y.permute(1, 0, 2, 3)
                x_ = x.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)  # T x (B x N) x F #torch.Size([12, 64, 1, 2])
                y_ = y.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)
                
            
                h_encode = self.model.forward_encoder(x_) #D*num_layers x (B x N) x F
                y_pred = self.model.forward(x_, y_, server_graph_encoding = graph_encoding)
                loss = nn.MSELoss()(y_pred, y_) # there seems to be questionable
                loss_feature = 0
                loss_feature += F.cosine_similarity(graph_encoding,h_encode,dim=-1).mean() #D*num_layers x (B x N)
                loss -= loss_feature
            
                loss.backward()
                self.optimizer.step()

                if step == self.E - 1: #这里的意思是说，轮到最后一个epoch的时候，每个batch的graph_encoding都会加进去（这是第二层for循环啊）  两层for循环啊，
                    features.append(h_encode.detach().clone().cpu()) #D*num_layers x (B x N) x F
     
                if get_learning_rate(self.optimizer) > 2e-6:
                    self.lr_scheduler.step()

                total_loss.append(float(loss))
                  
        if self.batch_adj:
            self.features = torch.stack(features) # [N_batch, B, F]
            #self.labels = torch.stack(labels) # [N_batch, B, C]
            #self.representations = torch.cat([self.features, self.labels],dim=-1)
        else:
            self.features = torch.cat(features, dim=1) #D*num_layers x (B x N) x F
            self.features = self.features.mean(dim=1) #D*num_layers x F
            self.features = self.features.mean(dim=0) # F
            #self.labels = torch.cat(labels, dim=0)
            #self.labels = self.labels.mean(dim=0)
            #self.representations = torch.cat([self.features, self.labels])
        return np.mean(total_loss)

    def test_shakespeare(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y,graph_encoding,_ in self.testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                test_samples += y.size(0)

                chunk_len = y.size(1)
                y_pred = self.model(x)

                loss += self.shakespeare_loss(y_pred, y).sum().detach() / chunk_len
                #test_acc += self.metric(y_pred, y).detach() / chunk_len
                test_acc += (torch.sum(torch.argmax(y_pred, dim=1) == y)).item()/ chunk_len

        return  test_acc, loss, test_samples
    
    def test(self):
        if 'shakespeare' in self.dataset: 
            return self.test_shakespeare()
        if 'metr-la' in self.dataset: 
            return self.test_metr_la()
        self.model.eval()
        test_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y, graph_encoding, _ in self.testloader:
                x, y, graph_encoding = x.to(self.device), y.to(self.device), graph_encoding.to(self.device)
                graph_encoding = graph_encoding.flatten(0, 1) # (B x N) x F
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:       
                    output=self.model(x)
                    loss+= self.ce_loss(output, y)
                else: 
                    output=self.model(x)['output']
                    loss+= self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_samples += y.shape[0]

        return test_acc, loss, test_samples
    
    def train_error_and_loss_shakespeare(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y,graph_encoding, _ in self.trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                test_samples += y.size(0)

                chunk_len = y.size(1)
                y_pred = self.model(x)

                loss += self.shakespeare_loss(y_pred, y).sum().detach() / chunk_len
                #test_acc += self.metric(y_pred, y).detach() / chunk_len
                train_acc += (torch.sum(torch.argmax(y_pred, dim=1) == y)).item()/ chunk_len

        return  train_acc, loss, test_samples
    
    def train_error_and_loss(self):
        if 'shakespeare' in self.dataset: 
            return self.train_error_and_loss_shakespeare()
        if 'metr-la' in self.dataset: 
            return self.train_error_and_loss_metr_la()
        self.model.eval()
        train_acc = 0
        loss = 0
        train_samples = 0
        with torch.no_grad():
            for x, y, graph_encoding, _ in self.trainloader:
                x, y, graph_encoding = x.to(self.device), y.to(self.device), graph_encoding.to(self.device)
                graph_encoding = graph_encoding.flatten(0, 1) # (B x N) x F
                    
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:     
                    output=self.model(x)
                    loss += self.ce_loss(output, y)
                else: 
                    output=self.model(x)['output']
                    loss += self.loss(output, y)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_samples += y.shape[0]
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
        return train_acc, loss , train_samples
    
    def test_metr_la(self):
        batch_size = self.batch_size
        self.model.eval()
        test_samples = 0
        total_loss = 0
        log = defaultdict(lambda : 0.0)
        with torch.no_grad():
            for x, y, graph_encoding, _ in self.testloader:
                
                test_samples += x.shape[0]
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

                graph_encoding = graph_encoding.permute(1, 0, 2, 3).flatten(1, 2).float().to(self.device)
                
                x = x.permute(1, 0, 2, 3)
                y = y.permute(1, 0, 2, 3)
                x_ = x.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)  # T x (B x N) x F #torch.Size([12, 64, 1, 2])
                y_ = y.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)
                
                y_pred = self.model.forward(x_, y_, server_graph_encoding = graph_encoding)
                loss = nn.MSELoss()(y_pred, y_) # there seems to be questionable
                total_loss+=loss


                feature_scaler = StandardScaler(
                    mean=54.4059283, std=19.49373927)
    
                metrics = self.unscaled_metrics(y_pred, y_, feature_scaler, 'test') # store each batch metric
                for k in metrics:
                    log[k] += (metrics[k] * x.shape[0]).cpu() # calculate the sum metrics of all batches 
                    #print(log['test/mape'])
                    
                   #for k in log:
                    #log[k] /= test_samples # calculate batch avg metric
                    #log[k] = log[k]
        return log, total_loss, test_samples

    def train_error_and_loss_metr_la(self):
        batch_size = self.batch_size
        self.model.eval()
        train_samples = 0
        total_loss = 0
        log = defaultdict(lambda : 0.0)
        with torch.no_grad():
            for x, y, graph_encoding, _ in self.trainloader:
                train_samples += x.shape[0]
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

                                # L x (B x N) x F
                graph_encoding = graph_encoding.permute(1, 0, 2, 3).flatten(1, 2).float().to(self.device) #  D*num_layers x (B x N) x F
                x = x.permute(1, 0, 2, 3)
                y = y.permute(1, 0, 2, 3)
                x_ = x.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)  # T x (B x N) x F #torch.Size([12, 64, 1, 2])
                y_ = y.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)
                y_pred = self.model.forward(x_, y_, server_graph_encoding = graph_encoding)
                loss = nn.MSELoss()(y_pred, y_) # there seems to be questionable
                total_loss+=loss


                feature_scaler = StandardScaler(
                    mean=54.4059283, std=19.49373927)
    
                metrics = self.unscaled_metrics(y_pred, y_, feature_scaler, 'train') # store each batch metric
                for k in metrics:
                    log[k] += (metrics[k] * x.shape[0]).cpu() # calculate the sum metrics of all batches 
            #for k in log:
                    #log[k] /= test_samples # calculate batch avg metric
                    #log[k] = log[k]
        return log, total_loss, train_samples
    
    def get_next_train_batch(self, count_labels=False):
        try:
            # Samples a new batch for personalizing
            (X, y, graph_encoding, _) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y,graph_encoding,  _) = next(self.iter_trainloader)
        result = {'X': X, 'y': y, 'graph_encoding':graph_encoding}
        '''
        if count_labels:
            unique_y, counts=torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        '''
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y, graph_encoding, _) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y, graph_encoding, _) = next(self.iter_testloader)
        result = {'X': X, 'y': y, 'graph_encoding':graph_encoding}
        return result
    
    def train(self, glob_iter=0, return_encode = False, return_grad = False, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        unseen = False
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True, return_encode = return_encode, return_grad = return_grad)

        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True, return_encode = return_encode, return_grad = return_grad)
                
    def criterion(self, loss, mode):
        #self.args.reg > 0 and mode != "PerTrain" and self.args.clients != 1:
        self.m1 = sd_matrixing(self.model.state_dict()).reshape(1, -1).to(self.device)
        self.m2 = sd_matrixing(self.server_param).reshape(1, -1).to(self.device)
        self.m3 = sd_matrixing(self.global_param).reshape(1, -1).to(self.device)
        self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
        self.reg2 = torch.nn.functional.pairwise_distance(self.m1, self.m3, p=2)
        loss = loss + 0.3 * self.reg1 + 0.3 * self.reg2    
        return loss

    def set_parameters_SFL(self, global_param, server_param):
        self.model.load_state_dict(global_param)
        self.global_param = global_param
        self.server_param = server_param
    