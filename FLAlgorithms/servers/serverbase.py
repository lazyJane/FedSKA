import torch
import os
import numpy as np
import h5py
from utils.model_utils import *
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from utils.model_utils import get_log_path, METRICS
from FLAlgorithms.users.userbase import User
from FLAlgorithms.users.userIFCA import UserIFCA
from tqdm import tqdm
from collections import defaultdict

class Server:
    def __init__(self, args, model, data_participate, data_unseen, seed):
        
        '''
        if not data_participate:
            a=1
            self.len_public = 0
        else:
        '''
        if 'SKA' in args.algorithm or 'Graph' in args.algorithm:
            print('SKA')
            self.train_datasets = data_participate[0]
            self.test_datasets = data_participate[1]
            self.train_iterators = data_participate[2]
            self.val_iterators = data_participate[3]
            self.test_iterators = data_participate[4]
            self.public_loader = data_participate[5]
            self.len_trains=data_participate[6] 
            self.len_tests=data_participate[7]
            self.len_public=data_participate[8]  
            self.n_classes=data_participate[9]  
        else:
            self.train_iterators = data_participate[0]
            self.val_iterators = data_participate[1]
            self.test_iterators = data_participate[2]
            self.public_loader = data_participate[3]
            self.len_trains=data_participate[4] 
            self.len_tests=data_participate[5]
            self.len_public=data_participate[6]  
            self.n_classes=data_participate[7]  
            
        #print(self.train_iterators)
        #print(self.val_iterators)
        #print(self.test_iterators)
        #print(self.public_loader)
        #print(self.len_trains)
        #print(self.len_tests)
        #print(self.len_public)  
        self.total_users=len(self.train_iterators)
        print("Users in total: {}".format(self.total_users))


        self.unseen_train_iterators = data_unseen[0]
        self.unseen_val_iterators = data_unseen[1]
        self.unseen_test_iterators = data_unseen[2]
        self.unseen_len_trains=data_unseen[4] 
        self.unseen_len_tests=data_unseen[5]
        self.unseen_total_users=len(self.unseen_train_iterators)
        self.unseen_total_train_samples=0
        self.unseen_users = []

        
        for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
            enumerate(tqdm(zip(self.unseen_train_iterators, self.unseen_val_iterators, self.unseen_test_iterators, self.unseen_len_trains, self.unseen_len_tests), total=len(self.unseen_train_iterators))):
            if train_iterator is None or test_iterator is None:
                continue
            #GPU_idx = task_id % 6
            #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
            #user_model = create_model_new(args, user_device)
            if args.algorithm == 'FedIFCA':
                user = UserIFCA(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, p=args.p, n_classes=self.n_classes, use_adam=False)
            else:
                user = User(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, n_classes=self.n_classes, use_adam=False)
            self.unseen_users.append(user)
            self.unseen_total_train_samples += user.train_samples
        
        #print("Users in total: {}".format(self.unseen_total_users))

        # Set up the main attributes
        self.dataset = args.dataset.lower()
        self.device =  args.device
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.K = args.K
        self.E = args.E
        if args.load_model == True:
            self.model = self.load_model()
        else: 
            self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        self.users = []

        self.selected_users = []
        self.num_users = args.num_users
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.personalized = 'pFed' in self.algorithm
        self.mode='partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.metrics = {key:[] for key in METRICS}
        self.timestamp = None
        self.save_path = args.result_path
        self.unseen_save_path = args.result_path + "unseen"
        os.system("mkdir -p {}".format(self.save_path))
        os.system("mkdir -p {}".format(self.unseen_save_path))


    def if_personalized(self):
        return 'pFed' in self.algorithm or 'PerAvg' in self.algorithm

    def if_ensemble(self):
        return 'FedE' in self.algorithm

    def send_parameters(self, users, mode='all', beta=1):
        for user in users:
            if mode == 'all': # share only subset of parameters
                user.set_parameters(self.model,beta=beta)
            else: # share all parameters
                user.set_shared_parameters(self.model,mode=mode)

    def send_p_parameters(self, models):
        users = self.selected_users
        '''
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        '''
        for user in users:
            user.set_p_parameters(models)

    def send_cluster_parameters(self, cluster_assign , mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        for i, user in enumerate(users):
            p = cluster_assign[i]
            if mode == 'all': # share only subset of parameters
                user.set_parameters(self.models[p],beta=beta)
            else: # share all parameters
                user.set_shared_parameters(self.model,mode=mode)


    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio



    def aggregate_parameters(self,partial=False):
        #if self.E != 0:
            #self.model.to(self.device)
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        if partial:
            for param in self.model.get_shared_parameters():#
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data) # 服务器模型参数重置为0
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples # 所有用户样本总数
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train,partial=partial)
    
    def personalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters())) # 上一轮全局模型参数
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train) # 模型参数聚合 self.model->server.model
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()): #上一轮的，聚合后的
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data

    def save_model(self, args):
        if self.E > 0:
            model_path = os.path.join("models", self.dataset, "E=" + str(self.E), args.algorithm)
        elif self.local_epochs > 0:
            model_path = os.path.join("models", self.dataset, "local_epochs=" + str(self.local_epochs), args.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))
    
    def save_model_center(self, args):
        if self.E > 0:
            model_path = os.path.join("models", self.dataset, "E=" + str(self.E), args.algorithm)
        elif self.local_epochs > 0:
            model_path = os.path.join("models", self.dataset, "local_epochs=" + str(self.local_epochs), args.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for i in range(self.p):
            torch.save(self.models[i], os.path.join(model_path, "center" + str(i) + ".pt"))

    def save_users_model(self, args):
        if self.E > 0:
            model_path = os.path.join("models", self.dataset, "E=" + str(self.E), args.algorithm)
        elif self.local_epochs > 0:
            model_path = os.path.join("models", self.dataset, "local_epochs=" + str(self.local_epochs), args.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for i, user in enumerate(self.users):
            torch.save(user.model, os.path.join(model_path, "user" + str(i) +".pt"))

    def save_cluster_assign(self, args):
        save_cluster_assign_path = "cluster_assign"
        
        alg = get_log_path(args, args.algorithm, self.seed, args.gen_batch_size) # EMnist-alpha1.0-ratio0.5_FedKLtopk_0.01_10u_32b_20_0
        os.makedirs(os.path.join(save_cluster_assign_path, alg), exist_ok=True)
        txt_path = os.path.join(save_cluster_assign_path, alg, "cluster.txt")

        cluster_assign = [] # 按用户顺序
        for user in self.users:
            cluster_assign.append(user.cluster_idx) 

        with open(txt_path, 'a') as f:
            f.write(str(cluster_assign)+'\n')

    def load_center_model(self):
        if self.E > 0:
            model_path = os.path.join("models", self.dataset, "E=" + str(self.E))
        elif self.local_epochs > 0:
            model_path = os.path.join("models", self.dataset, "local_epochs=" + str(self.local_epochs))
        for i in range(self.p):
            self.models[i] = torch.load(os.path.join(model_path, self.algorithm, "center" + str(i) + ".pt"))

    def load_model(self):
        if self.E > 0:
            model_path = os.path.join("models", self.dataset, "E=" + str(self.E), self.algorithm, "server" + ".pt")
        elif self.local_epochs > 0:
            model_path = os.path.join("models", self.dataset, "local_epochs=" + str(self.local_epochs), self.algorithm, "server" + ".pt")
        print(model_path)
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users #如果是全部用户，按顺序返回

        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)

    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=self.device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character] #第i个字符对应的label_weights
        labels_weight = labels_weight * 8
        self.shakespeare_loss = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(self.device)


    def save_results(self, args):
        #prefix = args.dataset
        alg = get_log_path(args, args.algorithm, self.seed, args.gen_batch_size)
        with h5py.File("./{}/{}.h5".format(self.save_path, alg), 'w') as hf: 
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()

    def save_unseen_results(self, args):
        #prefix = args.dataset
        alg = get_log_path(args, args.algorithm, self.seed, args.gen_batch_size)
        with h5py.File("./{}/{}.h5".format(self.unseen_save_path, alg), 'w') as hf: 
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()
        
    def train_error_and_loss(self, users):
        num_samples = []
        tot_correct = []
        losses = []
        for c in users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test(self, users, selected=False):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in users:
            ct, c_loss, ns = c.test() # test_acc, loss, y.shape[0]
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        #(losses)
        #input()ccc
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses
    
    

    def train_error_and_loss_personalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, c_loss, ns = c.train_error_and_loss_personalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_metr_la(self, users, selected=False):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_metrics = defaultdict(lambda : [])
        losses = []
        for c in users:
            cm, c_loss, ns = c.test_metr_la() # test_acc, loss, y.shape[0]
            for k in cm:
                tot_metrics[k].append(cm[k])
            num_samples.append(ns)
            losses.append(c_loss)
        #(losses)
        #input()ccc
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_metrics, losses
    
    def train_error_and_loss_metr_la(self, users):
        num_samples = []
        tot_metrics = defaultdict(lambda : [])
        losses = []
        for c in users:
            cm, c_loss, ns = c.train_error_and_loss_metr_la() # test_acc, loss, y.shape[0]
            for k in cm:
                tot_metrics[k].append(cm[k])
            num_samples.append(ns)
            losses.append(c_loss)
        #(losses)
        #input()ccc
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_metrics, losses
        

    def test_personalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, c_loss, ns = c.test_personalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses


    def evaluate(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        if 'metr-la' in self.dataset:
            return self.evaluate_metr_la()
        test_ids, test_samples, test_accs, test_losses = self.test(self.users)
        train_ids, train_samples, train_accs, train_losses = self.train_error_and_loss(self.users)
            
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        acc_users = np.array(test_accs) / np.array(test_samples)
        acc_users_idx = np.argsort(acc_users)
        acc_users= np.sort(acc_users)
        glob_acc_bottom_1 = acc_users[0]
        glob_acc_bottom_2 = acc_users[1]
        glob_acc_bottom_3 = acc_users[2]
        glob_acc_bottom_4 = acc_users[3]
        glob_acc_bottom_5 = acc_users[4]
        glob_acc_top_1 = acc_users[-1]
        glob_acc_top_2 = acc_users[-2]
        glob_acc_top_3 = acc_users[-3]
        glob_acc_top_4 = acc_users[-4]
        glob_acc_top_5 = acc_users[-5]
        glob_acc_bottom_idx_1 = acc_users_idx[0]
        glob_acc_bottom_idx_2 = acc_users_idx[1]
        glob_acc_bottom_idx_3 = acc_users_idx[2]
        glob_acc_bottom_idx_4 = acc_users_idx[3]
        glob_acc_bottom_idx_5 = acc_users_idx[4]
        print(glob_acc_bottom_idx_1,glob_acc_bottom_idx_2,glob_acc_bottom_idx_3,glob_acc_bottom_idx_4,glob_acc_bottom_idx_5)
        print(test_samples)
        print(test_losses)
        glob_loss = np.sum([torch.tensor(x * y).cpu().detach() for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)

        train_acc = np.sum(train_accs)*1.0/np.sum(train_samples)
        train_loss = np.sum([torch.tensor(x * y).cpu().detach() for (x, y) in zip(train_samples, train_losses)]).item() / np.sum(train_samples)
            #train_idcs, train_samples, train_accs, train_losses =
        
        if save:
                self.metrics['train_acc'].append(train_acc)
                self.metrics['train_loss'].append(train_loss)
                self.metrics['glob_acc'].append(glob_acc)
                self.metrics['glob_loss'].append(glob_loss)
                self.metrics['glob_acc_bottom_1'].append(glob_acc_bottom_1)
                self.metrics['glob_acc_bottom_2'].append(glob_acc_bottom_2)
                self.metrics['glob_acc_bottom_3'].append(glob_acc_bottom_3)
                self.metrics['glob_acc_bottom_4'].append(glob_acc_bottom_4)
                self.metrics['glob_acc_bottom_5'].append(glob_acc_bottom_5)
                self.metrics['glob_acc_top_1'].append(glob_acc_top_1)
                self.metrics['glob_acc_top_2'].append(glob_acc_top_2)
                self.metrics['glob_acc_top_3'].append(glob_acc_top_3)
                self.metrics['glob_acc_top_4'].append(glob_acc_top_4)
                self.metrics['glob_acc_top_5'].append(glob_acc_top_5)
        print("Average Train Accurancy = {:.4f}, Train Loss = {:.4f}, Average Test Accurancy = {:.4f}, Bottom accurancy = {:.4f}, Test Loss = {:.2f}.".format(train_acc, train_loss, glob_acc, glob_acc_bottom_1, glob_loss))

    def evaluate_metr_la(self, save=True, selected=False):
        test_ids, test_samples, test_metrics, test_losses = self.test_metr_la(self.users)
        train_ids, train_samples, train_metrics, train_losses = self.train_error_and_loss_metr_la(self.users)

            
        glob_mae = np.sum(test_metrics['test/mae'])*1.0 / np.sum(test_samples)
        glob_mse = np.sum(test_metrics['test/mse'])*1.0 / np.sum(test_samples)
        glob_mape = np.sum(test_metrics['test/mape'])*1.0 / np.sum(test_samples)
        mae_users = np.array(test_metrics['test/mae']) / np.array(test_samples)
        mse_users = np.array(test_metrics['test/mse']) / np.array(test_samples)
        mape_users =  np.array(test_metrics['test/mape']) / np.array(test_samples)
        glob_loss = np.sum([x * y.cpu().detach() for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)

        train_mae = np.sum(train_metrics['train/mae'])*1.0 / np.sum(train_samples)
        train_mse = np.sum(train_metrics['train/mse'])*1.0 / np.sum(train_samples)
        train_mape = np.sum(train_metrics['train/mape'])*1.0 / np.sum(train_samples)
        train_loss = np.sum([x * y.cpu().detach() for (x, y) in zip(train_samples, train_losses)]).item() / np.sum(train_samples)

        if save:
                self.metrics['train_mae'].append(train_mae)
                self.metrics['train_mse'].append(train_mse)
                self.metrics['train_mape'].append(train_mape)
                self.metrics['train_loss'].append(train_loss)
                self.metrics['glob_mae'].append(glob_mae)
                self.metrics['glob_mse'].append(glob_mse)
                self.metrics['glob_mape'].append(glob_mape)
                self.metrics['glob_loss'].append(glob_loss)

        print("Average Train MAE = {:.4f}, Average Train MSE = {:.4f}, Average Train MAPE = {:.4f}, Train Loss = {:.2f}, \n".format(train_mae, train_mse, train_mape, train_loss),
                  "Average Test MAE = {:.4f}, Average Test MSE = {:.4f}, Average Test MAPE = {:.4f}, Test Loss = {:.2f}, \n"
                  .format(glob_mae, glob_mse, glob_mape, glob_loss))    
    

    def evaluate_personalized_model(self,save=True, selected=False):
        ids, test_samples, test_accs, test_losses = self.test_personalized_model()  
        ids, train_samples, train_accs, train_losses = self.train_error_and_loss_personalized_model()
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        glob_acc_bottom = min(np.array(test_accs) / np.array(test_samples))
        glob_loss = np.sum([x * y.cpu().detach() for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)

        train_acc = np.sum(train_accs)*1.0/np.sum(train_samples)
        train_loss = sum([x * y.cpu().detach() for (x, y) in zip(train_samples, train_losses)]).item() / np.sum(train_samples)
        
        '''
        self.metrics['train_acc'].append(train_acc)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['glob_acc'].append(glob_acc)
        self.metrics['glob_loss'].append(glob_loss)
        self.metrics['glob_acc_bottom'].append(glob_acc_bottom)
        print("Average Train Accurancy = {:.4f}, Train Loss = {:.4f}, Average Test Accurancy = {:.4f}, Bottom accurancy = {:.4f}, Test Loss = {:.2f}.".format(train_acc, train_loss, glob_acc, glob_acc_bottom, glob_loss))
        '''
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        acc_users = np.array(test_accs) / np.array(test_samples)
        acc_users_idx = np.argsort(acc_users)
        acc_users= np.sort(acc_users)
        glob_acc_bottom_1 = acc_users[0]
        glob_acc_bottom_2 = acc_users[1]
        glob_acc_bottom_3 = acc_users[2]
        glob_acc_bottom_4 = acc_users[3]
        glob_acc_bottom_5 = acc_users[4]
        glob_acc_top_1 = acc_users[-1]
        glob_acc_top_2 = acc_users[-2]
        glob_acc_top_3 = acc_users[-3]
        glob_acc_top_4 = acc_users[-4]
        glob_acc_top_5 = acc_users[-5]
        glob_acc_bottom_idx_1 = acc_users_idx[0]
        glob_acc_bottom_idx_2 = acc_users_idx[1]
        glob_acc_bottom_idx_3 = acc_users_idx[2]
        glob_acc_bottom_idx_4 = acc_users_idx[3]
        glob_acc_bottom_idx_5 = acc_users_idx[4]
        print(glob_acc_bottom_idx_1,glob_acc_bottom_idx_2,glob_acc_bottom_idx_3,glob_acc_bottom_idx_4,glob_acc_bottom_idx_5)
        glob_loss = np.sum([x * y.cpu().detach() for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)

        train_acc = np.sum(train_accs)*1.0/np.sum(train_samples)
        train_loss = np.sum([x * y.cpu().detach() for (x, y) in zip(train_samples, train_losses)]).item() / np.sum(train_samples)
        #train_idcs, train_samples, train_accs, train_losses =
       
        if save:
            self.metrics['train_acc'].append(train_acc)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
            self.metrics['glob_acc_bottom_1'].append(glob_acc_bottom_1)
            self.metrics['glob_acc_bottom_2'].append(glob_acc_bottom_2)
            self.metrics['glob_acc_bottom_3'].append(glob_acc_bottom_3)
            self.metrics['glob_acc_bottom_4'].append(glob_acc_bottom_4)
            self.metrics['glob_acc_bottom_5'].append(glob_acc_bottom_5)
            self.metrics['glob_acc_top_1'].append(glob_acc_top_1)
            self.metrics['glob_acc_top_2'].append(glob_acc_top_2)
            self.metrics['glob_acc_top_3'].append(glob_acc_top_3)
            self.metrics['glob_acc_top_4'].append(glob_acc_top_4)
            self.metrics['glob_acc_top_5'].append(glob_acc_top_5)
        print("Average Train Accurancy = {:.4f}, Train Loss = {:.4f}, Average Test Accurancy = {:.4f}, Bottom accurancy = {:.4f}, Test Loss = {:.2f}.".format(train_acc, train_loss, glob_acc, glob_acc_bottom_1, glob_loss))

    def evaluate_ensemble(self, selected=True):
        self.model.eval()
        users = self.selected_users if selected else self.users
        test_acc=0
        loss=0
        for x, y in self.testloaderfull:
            target_logit_output=0
            for user in users:
                # get user logit
                user.model.eval()
                user_result=user.model(x, logit=True)
                target_logit_output+=user_result['logit']
            target_logp=F.log_softmax(target_logit_output, dim=1)
            test_acc+= torch.sum( torch.argmax(target_logp, dim=1) == y ) #(torch.sum().item()
            loss+=self.loss(target_logp, y)
        loss = loss.detach().numpy()
        test_acc = test_acc.detach().numpy() / y.shape[0]
        self.metrics['glob_acc'].append(test_acc)
        self.metrics['glob_loss'].append(loss)
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))

    def unseen_user_train(self, unseen_E, unseen_local_epochs):
        for user in self.unseen_users:
            if unseen_E > 0: 
                user.E = unseen_E
                user.fit_epochs(lr_decay=True, unseen = True)
            if unseen_local_epochs > 0: 
                user.local_epochs = unseen_local_epochs
                user.fit_batches(lr_decay=True, unseen = True)

    def update_unseen_users(self, unseen_E = 0, unseen_local_epochs = 0):
        self.send_parameters(self.unseen_users)
        
        #for glob_iter in range(5):
            #for user in self.unseen_users:
                #ser.train_unseen(glob_iter)
                   
    def evaluate_unseen_users(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(self.unseen_users)
        train_ids, train_samples, train_accs, train_losses = self.train_error_and_loss(self.unseen_users)
        
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        glob_acc_bottom = min(np.array(test_accs) / np.array(test_samples))
        glob_acc_bottom_idx = np.argmin(np.array(test_accs) / np.array(test_samples))
        print(glob_acc_bottom_idx)
        glob_loss = np.sum([x * y.cpu().detach() for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)

        train_acc = np.sum(train_accs)*1.0/np.sum(train_samples)
        train_loss = np.sum([x * y.cpu().detach() for (x, y) in zip(train_samples, train_losses)]).item() / np.sum(train_samples)
       
        if save:
            self.metrics['unseen_train_acc'].append(train_acc)
            self.metrics['unseen_train_loss'].append(train_loss)
            self.metrics['unseen_glob_acc'].append(glob_acc)
            self.metrics['unseen_glob_loss'].append(glob_loss)
            self.metrics['unseen_glob_acc_bottom'].append(glob_acc_bottom)
        print("Average Unseen Train Accurancy = {:.4f}, Unseen Train Loss = {:.4f}, Average Unseen Test Accurancy = {:.4f}, Unseen Bottom accurancy = {:.4f}, Unseen Test Loss = {:.2f}.".format(train_acc, train_loss, glob_acc, glob_acc_bottom, glob_loss))

    