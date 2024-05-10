from FLAlgorithms.users.userFedProx import UserFedProx
from FLAlgorithms.servers.serverbase import *
# Implementation for FedProx Server
from tqdm import tqdm
class FedProx(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)
        
        if args.test_unseen == True:
            a = 1
        else:
            for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
                enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
                if train_iterator is None or test_iterator is None:
                    continue
                #GPU_idx = total_users % 6
                #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
                user = UserFedProx(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, self.n_classes,  use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating FedProx server.")

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            if glob_iter == 0: self.send_parameters(self.users, mode=self.mode)
            
            for user in self.selected_users: # allow selected users to train
                    user.train(glob_iter)
            self.aggregate_parameters()
            self.send_parameters(self.users, mode=self.mode)
            self.evaluate()
            
            self.save_results(args)
        self.save_model(args)