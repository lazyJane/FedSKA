from fcntl import F_SETLEASE
import torch
import torch.nn as nn
from FLAlgorithms.users.userbase import User
import copy
from FLAlgorithms.optimizers.fedoptimizer import PerturbedGradientDescent
from utils.model_utils import *

class UserDitto(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes,  use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False)
         
        self.pmodel = copy.deepcopy(self.model)
        
        if 'metr-la' in self.dataset:
            self.poptimizer = torch.optim.Adam(self.pmodel.parameters(), lr=self.learning_rate)
        elif 'mnist' == self.dataset[0]:
            self.poptimizer = torch.optim.SGD(self.pmodel.parameters(), lr=self.learning_rate)
        else:  
            self.poptimizer = torch.optim.SGD(self.pmodel.parameters(), lr=self.learning_rate,  momentum=0.9)
        #self.poptimizer = PerturbedGradientDescent(
            #self.pmodel.parameters(), lr=self.learning_rate, mu=0.01)
        self.plr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.poptimizer, gamma=0.99) # 学习率指数衰减
        

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)
                
    def ptrain_shakespeare(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        unseen = False
        if self.E != 0: 
            self.pmodel.train()
            for step in range(self.E):
                for x, y, indices in self.trainloader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                # n_samples += y.size(0)
                    chunk_len = y.size(1)
                    self.poptimizer.zero_grad()

                    y_pred = self.pmodel(x)
                    loss_vec = self.shakespeare_loss(y_pred, y)

                    #if weights is not None:
                        #weights = weights.to(self.device)
                        #loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
                    #else:
                    loss = loss_vec.mean()

                    loss.backward()
                    self.poptimizer.step()
                if lr_decay:
                    if unseen:
                        self.plr_scheduler.step(step)
                    else:
                        self.plr_scheduler.step(glob_iter) #存疑
        else:
            self.pmodel.train()
            for epoch in range(1, self.local_epochs + 1):
                for i in range(self.K):
                    result =self.get_next_train_batch(count_labels=count_labels)
                    x, y = result['X'], result['y']
                    x, y = x.to(self.device), y.to(self.device)
                    #if count_labels:
                        #self.update_label_counts(result['labels'], result['counts'])
                    self.poptimizer.zero_grad()
                    chunk_len = y.size(1)
                    self.optimizer.zero_grad()

                    y_pred = self.pmodel(x)
                    loss_vec = self.shakespeare_loss(y_pred, y)

                    #if weights is not None:
                        #weights = weights.to(self.device)
                        #loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
                    #else:
                    loss = loss_vec.mean()
               
                    loss.backward()
                    self.poptimizer.step(self.pmodel.parameters())#self.plot_Celeb)
            if lr_decay:
                if unseen:
                        self.plr_scheduler.step(epoch)
                else:
                        self.plr_scheduler.step(glob_iter) #存疑

    def ptrain_metr_la(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        unseen = False
        batch_size = self.batch_size
        if self.E != 0: 
            self.pmodel.train()
            for step in range(self.E):
                for x, y, _ in self.trainloader: 
                #print(x.shape)# [64,12,1,2]  # B x T x N x F
                #print(y.shape)
                #input()
                    self.poptimizer.zero_grad()
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

                    x = x.permute(1, 0, 2, 3)
                    y = y.permute(1, 0, 2, 3)
                    x_ = x.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)  # T x (B x N) x F #torch.Size([12, 64, 1, 2])
                    y_ = y.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)
                    y_pred = self.pmodel(x_, y_)
                    loss = nn.MSELoss()(y_pred, y_) # there seems to be questionable

                
                    loss.backward()
                    self.poptimizer.step()
                    
                if lr_decay:
                    if unseen:
                        self.plr_scheduler.step(step)
                    else:
                        #if get_learning_rate(self.optimizer) > 2e-6:
                        self.plr_scheduler.step(glob_iter) #存疑
        else:
            self.pmodel.train()
            for epoch in range(1, self.local_epochs + 1):
                for i in range(self.K):
                    result =self.get_next_train_batch(count_labels=count_labels)
                    x, y = result['X'], result['y']
                    x, y = x.to(self.device), y.to(self.device)
                    #if count_labels:
                        #self.update_label_counts(result['labels'], result['counts'])
                    self.poptimizer.zero_grad()
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

                    x = x.permute(1, 0, 2, 3)
                    y = y.permute(1, 0, 2, 3)
                    x_ = x.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)  # T x (B x N) x F #torch.Size([12, 64, 1, 2])
                    y_ = y.reshape(x.shape[0], -1, x.shape[3]).float().to(self.device)
                    y_pred = self.pmodel(x_, y_)
                    loss = nn.MSELoss()(y_pred, y_) # there seems to be questionable

                    
                    loss.backward()
                    self.poptimizer.step()
            if lr_decay:
                if unseen:
                        self.plr_scheduler.step(epoch)
                else:
                        self.plr_scheduler.step(glob_iter) #存疑

    def criterion(self, loss):
        #self.args.reg > 0 and mode != "PerTrain" and self.args.clients != 1:
        self.m1 = sd_matrixing(self.pmodel.state_dict()).reshape(1, -1).to(self.device)
        self.m2 = sd_matrixing(self.model.state_dict()).reshape(1, -1).to(self.device)
        self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
        loss = loss + 0.3 * self.reg1
        return loss
    
    def ptrain(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        unseen = False
        if self.E != 0: 
            # self.fit_epochs(glob_iter, lr_decay=True)
            if 'shakespeare' in self.dataset: 
                self.ptrain_shakespeare(glob_iter, lr_decay)
                return 
            elif 'metr' in self.dataset: 
                self.ptrain_metr_la(glob_iter, lr_decay)
                return 
            self.model.train()
            self.pmodel.train()

            for step in range(self.E):
                for x, y, _ in self.trainloader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    self.poptimizer.zero_grad()
                    if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                        output=self.pmodel(x)
                        loss=self.ce_loss(output, y)
                    else:
                        output=self.pmodel(x)['output']
                        loss=self.loss(output, y)
                    loss = self.criterion(loss) 
                    loss.backward()
                    #self.poptimizer.step()
                    self.poptimizer.step()
                if lr_decay:
                    if unseen:
                        self.plr_scheduler.step(step)
                    else:
                        self.plr_scheduler.step(glob_iter) #存疑
        else:
            self.pmodel.train()
            for epoch in range(1, self.local_epochs + 1):
                for i in range(self.K):
                    result =self.get_next_train_batch(count_labels=count_labels)
                    X, y = result['X'], result['y']
                    X, y = X.to(self.device), y.to(self.device)
                    #if count_labels:
                        #self.update_label_counts(result['labels'], result['counts'])

                    self.optimizer.zero_grad()
                    if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                        output=self.pmodel(X)
                        loss=self.ce_loss(output, y)
                    else:
                        output=self.pmodel(X)['output']
                        loss=self.loss(output, y)
                    loss.backward()
                    self.poptimizer.step()#self.plot_Celeb)
            if lr_decay:
                if unseen:
                        self.plr_scheduler.step(epoch)
                else:
                        self.plr_scheduler.step(glob_iter) #存疑


    def test(self):
        if 'shakespeare' in self.dataset: 
            return self.test_shakespeare()
        self.pmodel.eval()
        test_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y,_ in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:       
                    output = self.pmodel(x)
                    loss += self.ce_loss(output, y)
                else: 
                    output = self.pmodel(x)['output']
                    loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_samples += y.shape[0]
        return test_acc, loss, test_samples

    def test_shakespeare(self):
        self.pmodel.eval()
        test_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y,_ in self.testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                test_samples += y.size(0)

                chunk_len = y.size(1)
                y_pred = self.pmodel(x)

                loss += self.shakespeare_loss(y_pred, y).sum().detach() / chunk_len
                #test_acc += self.metric(y_pred, y).detach() / chunk_len
                test_acc += (torch.sum(torch.argmax(y_pred, dim=1) == y)).item()/ chunk_len

        return  test_acc, loss, test_samples
                    
    def train_error_and_loss(self):
        if 'shakespeare' in self.dataset: 
            return self.train_error_and_loss_shakespeare()
        self.pmodel.eval()
        train_acc = 0
        loss = 0
        train_samples = 0
        with torch.no_grad():
            for x, y, _ in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:     
                    output = self.pmodel(x)
                    loss += self.ce_loss(output, y)
                else: 
                    output = self.pmodel(x)['output']
                    loss += self.loss(output, y)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_samples += y.shape[0]
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
        return train_acc, loss , train_samples

    def train_error_and_loss_shakespeare(self):
        self.pmodel.eval()
        train_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y,_ in self.trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                test_samples += y.size(0)

                chunk_len = y.size(1)
                y_pred = self.pmodel(x)

                loss += self.shakespeare_loss(y_pred, y).sum().detach() / chunk_len
                #test_acc += self.metric(y_pred, y).detach() / chunk_len
                train_acc += (torch.sum(torch.argmax(y_pred, dim=1) == y)).item()/ chunk_len

        return  train_acc, loss, test_samples
    
                  