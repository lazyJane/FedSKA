"""
Download CIFAR-10 dataset, and splits it among clients
"""
from sklearn.model_selection import train_test_split
import os
import argparse
import pickle

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset


from split_utils import *


ALPHA = .4
N_CLASSES = 10
N_COMPONENTS = 3
SEED = 12345
RAW_DATA_PATH = "raw_data/"


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_users',
        help='number of users;',
        type=int,
        default=20)
        #required=True)
    parser.add_argument(
        '--split',
        help='non_iid split type',
        type=str,
        default='dirichlet_non_iid_split'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 5',
        type=int,
        default=5
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters;',
        type=int,
        default=N_COMPONENTS
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;',
        type=float,
        default=ALPHA
    )
    parser.add_argument(
        '--edge_frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--val_frac',
        help='fraction of validation set (from train set); default: 0.0;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--unseen_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=SEED
    )

    return parser.parse_args()


def main():
    args = parse_args()

    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = CIFAR10(root=RAW_DATA_PATH, download=True, train=True, transform=transform)
    test_data = CIFAR10(root=RAW_DATA_PATH, download=False, train=False, transform=transform)
    #train_data (50000, 32, 32, 3) 
    #test_data (10000, 32, 32, 3)
    print(train_data.data.shape) #(50000, 32, 32, 3)
    #torch.save((train_data.data, train_data.targets), os.path.join("train.pt"))
    #torch.save((test_data.data, test_data.targets), os.path.join("test.pt"))
    #input()
    dataset =\
        ConcatDataset([
            CIFAR10(root=RAW_DATA_PATH, download=True, train=True, transform=transform),
            CIFAR10(root=RAW_DATA_PATH, download=False, train=False, transform=transform)
        ])
    #a = CIFAR10(root=RAW_DATA_PATH, download=True, train=True, transform=transform)
    #print(a.targets)
    #print(set(a.targets.numpy()))
    #input()
    dataset_indices = list(range(len(dataset)))
    public_indices = []
    label2index = rrangebyclass(dataset, N_CLASSES)
    for key, value in label2index.items():
        '''
        print(key)
        if key < 1: public_indices += value[0:1] # public从每类中取的数据量
        #print(len(label2index[key]))
        label2index[key] = value[2:]
        '''
        public_indices += value[0:10] # public从每类中取的数据量
        #print(len(label2index[key]))
        label2index[key] = value[10:]
    
    users_indices = list(set(dataset_indices) - set(public_indices)) #按序
    
    if args.split == 'split_iid':
        user_indices = split_iid(
            users_indices, 
            args.n_users, 
            args.s_frac, 
            seed = args.seed
            )
    # p(y)

    elif args.split == 'allocation_shards':
        user_indices, A = allocation_shards(
            indices = users_indices, 
            n_classes = N_CLASSES, 
            dataset = dataset,
            n_users = args.n_users,
            shards = args.shards,
            edge_frac=args.edge_frac,
            seed = args.seed
        )
    elif args.split == 'pathological_non_iid_split': 
        user_indices = pathological_non_iid_split(
            indices = users_indices, 
            n_classes = N_CLASSES, 
            dataset = dataset,
            n_users = args.n_users,
            n_classes_per_user = args.n_shards,
            frac=args.s_frac,
            seed = args.seed
            )
     # p(y)
    elif args.split == 'dirichlet_non_iid_split':
        user_indices = dirichlet_non_iid_split(
            indices = users_indices, 
            n_users = args.n_users, 
            dataset = dataset,
            n_classes = N_CLASSES, 
            alpha = args.alpha, 
            frac = args.s_frac, 
            n_clusters = args.n_components,
            seed = args.seed
            )
     # p(y|x)
    elif args.split == 'label_swapped_non_iid_split':
        user_indices =label_swapped_non_iid_split(
            indices = users_indices, 
            dataset = dataset,
            n_users = args.n_users, 
            frac = args.s_frac,
            k = args.n_components,
            seed = args.seed
            )
    elif args.split == 'rotated_non_iid_split':
        return #

    if args.unseen_tasks_frac > 0:
        participate_users_indices, unseen_users_indices = \
            train_test_split(
                user_indices, 
                test_size=args.unseen_tasks_frac, 
                random_state=args.seed)
        
    else:
        participate_users_indices, unseen_users_indices = user_indices, []

    if args.split == 'split_iid':
        split = 'iid'
    elif args.split == 'allocation_shards':
        split = 'allocation_shards'+str(args.shards)
    elif args.split == 'dirichlet_non_iid_split':
        split = 'alpha'+str(args.alpha)
    elif args.split == 'pathological_non_iid_split':
        split = 'pathological'
    elif args.split == 'label_swapped_non_iid_split':
        split = 'label_swapped'
    elif args.split == 'rotated_non_iid_split':
        split = 'rotated'

    path_prefix = f'u{args.n_users}-{split}-ratio{args.s_frac}'

    os.makedirs(os.path.join(path_prefix, "participate"), exist_ok=True)
    os.makedirs(os.path.join(path_prefix, "unseen"), exist_ok=True)

    if args.split == 'allocation_shards':
        A_path = os.path.join(path_prefix, "A.pkl")
        save_data(A, A_path)

    for mode, users_indices in [('participate', participate_users_indices), ('unseen', unseen_users_indices)]:
       
        for user_id, indices in enumerate(users_indices):
            user_path = os.path.join(path_prefix, mode, "uers_{}".format(user_id))
            os.makedirs(user_path, exist_ok=True)

            train_indices, test_indices =\
                train_test_split(
                    indices,
                    train_size=args.tr_frac,
                    random_state=args.seed
                )

            if args.val_frac > 0:
                train_indices, val_indices = \
                    train_test_split(
                        train_indices,
                        train_size=1.-args.val_frac,
                        random_state=args.seed
                    )
                save_data(val_indices, os.path.join(user_path, "val.pkl"))

            save_data(train_indices, os.path.join(user_path, "train.pkl"))
            save_data(test_indices, os.path.join(user_path, "test.pkl"))
            save_data(public_indices, os.path.join(path_prefix, "public.pkl"))

if __name__ == "__main__":
    main()
