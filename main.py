#!/usr/bin/env python
from utils.plot_utils import *
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverlocal import Fedlocal
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedCluster import FedCluster
from FLAlgorithms.servers.serverIFCA import FedIFCA
from FLAlgorithms.servers.serverFedSEM import FedSEM
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverFedDS import FedDS
from FLAlgorithms.servers.serverditto import FedDitto 
from FLAlgorithms.servers.serverSFL import FedSFL
from FLAlgorithms.servers.serverFedU import FedU
from FLAlgorithms.servers.serverSKA import FedSKA
from utils.model_utils import *
from utils.data_utils import *
from utils.constants import *
import torch
from multiprocessing import Pool

def build_knn_neighbourhood(attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
        return weighted_adjacency_matrix

def get_A(args): #normalized A
    data_dir_u = get_data_dir(args)
    print(data_dir_u)
    print("metr-la" in args.dataset)
    if "allocation" in args.dataset:
        print("allocation_adj_load")
        A_path = os.path.join(data_dir_u[0], "A.pkl")
        with open(A_path, "rb") as f:
            A = pickle.load(f)
    elif "metr-la" in args.dataset.lower():
        A_path = os.path.join(data_dir_u, "A.pkl")
        with open(A_path, "rb") as f:
            A = pickle.load(f)
    else:
        A = torch.eye(args.num_users)
    return A

def init_data(args, mode):
    #除了FEMNIST和Shakespeare都会返回类似 ('data/Cifar10/u80-alpha0.5-ratio1.0', '80')
    data_dir_u = get_data_dir(args)
    if len(data_dir_u) == 2: 
        data_dir = data_dir_u[0] 
    else: 
        data_dir = data_dir_u
    root_path=os.path.join(data_dir , mode)

    
    #root_path=os.path.join(data_dir , "participate")
    #root_path=os.path.join(data_dir , "unseen")
    u = 0 #用在label_swapped的时候，是participate的user数目
    if 'label_swapped' in args.dataset.lower():
        split = 'label_swapped'
        u =  int(data_dir_u[1])
    else: 
        split = '' #null表示其他

    #metr_la_path = 'data/traffic/data/METR-LA'
    #normalize = {}
    if 'metr-la' in args.dataset.lower():
        #feature_mean, feature_std = np.load(os.path.join(metr_la_path, 'train.npz'))
        #normalize['feature_mean'] = feature_mean
        #normalize['feature_std'] = feature_std
        experiment = 'metr-la'
    if 'femnist' in args.dataset.lower(): #Cifar100-alpha0.5-ratio1.0-u100
        experiment = 'femnist'
    elif 'vehicle_sensor' in args.dataset.lower(): #Cifar100-alpha0.5-ratio1.0-u100
        experiment = 'vehicle_sensor'
    elif 'emnist' in args.dataset.lower():
        experiment = 'emnist'
    elif 'mnist' in args.dataset.lower():
        experiment = 'mnist'
    
    elif 'cifar100' in args.dataset.lower():
        experiment = 'cifar100'
    elif 'cifar10' in args.dataset.lower():
        experiment = 'cifar10' 
    elif 'shakespeare' in args.dataset.lower():
        experiment = 'shakespeare'
    n_classes = N_CLASSES[experiment]
    print("===> Building data iterators..")
    print("u", u)
    print(LOADER_TYPE[experiment])
    if 'SKA' in args.algorithm or 'Graph' in args.algorithm:
        train_dataset, test_dataset, train_iterators, val_iterators, test_iterators , public_iterator, len_trains, len_tests, len_public=\
            get_loaders(
                args,
                split, 
                u,
                type_=LOADER_TYPE[experiment], #数据类型
                root_path=root_path, #  参与训练的数据目录
                data_dir=data_dir, # 总数据目录
                batch_size=args.batch_size,
                is_validation=False
            )
        #print(train_dataset, test_dataset, train_iterators, val_iterators, test_iterators, public_iterator, len_trains, len_tests, len_public)#, normalize)
        return train_dataset, test_dataset, train_iterators, val_iterators, test_iterators, public_iterator, len_trains, len_tests, len_public, n_classes#, normalize
    else:
        train_iterators, val_iterators, test_iterators , public_iterator, len_trains, len_tests, len_public=\
            get_loaders(
                args,
                split, 
                u,
                type_=LOADER_TYPE[experiment], #数据类型
                root_path=root_path, #  参与训练的数据目录
                data_dir=data_dir, # 总数据目录
                batch_size=args.batch_size,
                is_validation=False
            )
        return train_iterators, val_iterators, test_iterators, public_iterator, len_trains, len_tests, len_public, n_classes#, normalize
    
def build_graph(A):
    start_idx = []
    end_idx = []
                    
    for (idx_i,link_i) in enumerate(A):
        idx_i = torch.tensor(idx_i).type(torch.long)
        start_idx.append(idx_i)
        end_idx.append(idx_i)
        for idx_j in link_i:
            idx_j = torch.tensor(idx_j).type(torch.long)
            #self.A[i,j] = torch.tensor(self.A[i,j]).type(torch.long)
            #print(self.A[i,j])
                
            if int(A[idx_i,idx_j]) > 0:
                start_idx.append(idx_i)
                end_idx.append(idx_j)

    g = dgl.graph((start_idx, end_idx)) # 
    return g

def create_server_n_user(args, i):
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
        
    model = create_model_new(args)

    if 'SKA' in args.algorithm or 'FedSFL' in args.algorithm or 'FedU' in args.algorithm:
        if args.dataset == "vehicle_sensor":
            b = np.random.uniform(0,1,size=(args.num_users,args.num_users))
            b_symm = (b + b.T)/2 #对称矩阵
            b_symm[b_symm < 0.25] = 0 #
            A = b_symm
        else:
            A = get_A(args)
            #print(A)
            #input()
        #A = build_knn_neighbourhood(A, 30, 0)
        A = normalize_adj(A)

        if 'metr-la' in args.dataset.lower():
            g = METR_LAGraphDataset()
        else:
            g = build_graph(A)
            

    #args.device = server_device = "cpu"
    data_participate = init_data(args, mode = "participate") 
    data_unseen = init_data(args, mode = "unseen")
   
    if ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, data_participate, data_unseen, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, data_participate, data_unseen, i)
    elif ('FedDS' in args.algorithm):
        server = FedDS(args, model, data_participate, data_unseen, i)       
    elif ('FedCluster' in args.algorithm):
        server = FedCluster(args, model, data_participate, data_unseen, i)
    elif ('FedIFCA' in args.algorithm):
        server = FedIFCA(args, model, data_participate, data_unseen, i)
    elif('Fedlocal' in args.algorithm):
        server = Fedlocal(args, model, data_participate, data_unseen, i)
    elif('FedSEM' in args.algorithm):
        server = FedSEM(args, model, data_participate, data_unseen, i)
    
    elif('PerFedavg' in args.algorithm):
        server = PerAvg(args, model, data_participate, data_unseen, i)
    elif('pFedME' in args.algorithm):
        server = pFedMe(args, model, data_participate, data_unseen, i)
    elif('Ditto' in args.algorithm):
         server = FedDitto(args, model, data_participate, data_unseen, i)
    elif('FedU' in args.algorithm):
         server = FedU(args, model, data_participate, data_unseen, A, i)
    elif('FedSFL' in args.algorithm):
         server = FedSFL(args, model, data_participate, data_unseen, A, i)
    elif('FedSKA' in args.algorithm):
        server = FedSKA(args, model, data_participate, data_unseen, A, g, i)


    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server
    
def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        #server.test()

def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="pFedMe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    #parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=float, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
   # parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=0)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--p", type=int, default=3, help="CLuster numbers")
    #parser.add_argument('--w_decay', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--load_model", type=bool, default=False, help="Load model or train from start")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--E", type=int, default="0", help="fit_epochs or fit_batchs")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--test_unseen", type=bool, default=False)
    

    # adj learning
    # common
    #parser.add_argument('--k', type=int, default=98) #subgraph_size
    parser.add_argument('--k', type=int, default=98, help='k for initializing with knn')
    parser.add_argument('--epochs_adj', type=int, default=1, help='Number of epochs to learn the adjacency.')
    parser.add_argument('--lr_adj', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--w_decay_adj', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--user_feature_hidden_size', type=int, default=64) #client feature embedding dim 
    parser.add_argument('--hidden_adj', type=int, default=64, help='Number of hidden units.') #adj_learn hidden_size
    parser.add_argument('--dropout1', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropout2', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropout_adj1', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropout_adj2', type=float, default=0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--sparse', type=int, default=0)
    parser.add_argument('--base', type=str, default='feature') #feature_label
    parser.add_argument('--batch_adj', type=bool, default=False)


    # Structure Bootstrapping
    #parser.add_argument('--boot_strap', type=bool, default=False)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--c', type=int, default=0)
    parser.add_argument('--adjbeta', type=float, default=0.05, help='update ratio') # SFL 1-tau
    # IDGL
    parser.add_argument('--graph_learn_hidden_size', type=int, default=84)# kernel: 100, attention: 70  
    parser.add_argument('--graph_learn_epsilon', type=int, default=1.0)# weighted_cosine: 0
    parser.add_argument('--graph_learn_num_pers', type=int, default=10)
    parser.add_argument('--graph_metric_type', type=str, default='gat_attention', choices=["weighted_cosine", "kernel", "attention", "gat_attention"]) 
    
    # SLAP
    # adj leanrer
    
    parser.add_argument('--knn_metric', type=str, default='cosine', help='See choices', choices=['cosine', 'minkowski'])
    parser.add_argument('--i', type=int, default=6)
    parser.add_argument('--gen_mode', type=int, default=1)
    parser.add_argument('--nlayers_adj', type=int, default=3)
    parser.add_argument('--non_linearity', type=str, default='relu') #'relu'
    parser.add_argument('--mlp_act', type=str, default='relu', choices=["relu", "tanh"])
    parser.add_argument('--mlp_h', type=int, default=64) #cifar10 1280
    parser.add_argument('--mlp_epochs', type=int, default=100)
    parser.add_argument('--normalization', type=str, default='sym')

    # self-supervison loss
    parser.add_argument('--ratio', type=int, default=100, help='ratio of ones to select for each mask')
    parser.add_argument('--nr', type=int, default=5, help='ratio of zeros to ones')
    parser.add_argument('--noise', type=str, default="normal", choices=['mask', 'normal'])
    parser.add_argument('--loss', type=str, default="mse", choices=['mse', 'bce'])
    
    # graph-based aggregate
    parser.add_argument('--Graph_hidden_size', type=int, default=64) #cifar101280 mnist 64
    #parser.add_argument("--GAT_layers", type=int, default=3)
    parser.add_argument("--gen_layers", type=int, default=1)
    

    # SFL args.k = 30
    parser.add_argument("--adjalpha", type=float, default=3) # adj learning
    parser.add_argument("--agg", type=str, default="graph_v2")
    parser.add_argument("--serveralpha", type=float, default=1)
    parser.add_argument("--node_dim", type=int, default=40)

    # CNFGNN
    parser.add_argument("--server_epoch", type=int, default=10) # CNFGNN GCN learning epoch
    parser.add_argument('--server_lr', type=float, default=0.01)
    parser.add_argument('--server_weight_decay', type=float, default=0.01)
    parser.add_argument('--feature_hidden_size', type=int, default=64) #cifar101280 mnist 64
    parser.add_argument('--gru_num_layers', type=int, default=2)

    # SUBLIME
    # GSL Module
    parser.add_argument('--gsl_mode', type=str, default="structure_refinement",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('--type_learner', type=str, default='mlp', choices=["fgp", "att", "mlp", "gnn"])
    #parser.add_argument('--k', type=int, default=99) #knn建图
    parser.add_argument('--sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # GCL Module - Framework
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--w_decay', type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--rep_dim', type=int, default=64)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--contrast_batch_size', type=int, default=0)
    parser.add_argument('--nlayers', type=int, default=1)

    # GCL Module -Augmentation
    parser.add_argument('--maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('--maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('--dropedge_rate', type=float, default=0.0)

    # FedAMP
    parser.add_argument("--alphaK", type=float, default=5e-3)
    parser.add_argument("--sigma", type=float, default=1e-1)
    #parser.add_argument("--K", type=float, default=0.004, help="Computation steps")
    
    # FedU
    parser.add_argument("--L_K", type=float, default=0.1, help="Regularization term")
    

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    #print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)
    main(args)
