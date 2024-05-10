from xml.dom.minidom import Identified
import numpy as np
import scipy.sparse as sp
import random
import time
import pandas as pd
import torch
#from utils.data_utils import *
mapp=np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])

def rrangebyclass(dataset, n_classes):
    data_indices = list(range(len(dataset)))
    label2index = {k:[] for k in range(n_classes)}
    for idx in data_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)
    return label2index

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g) # 每组分到的大小
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist

def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res

def split_iid(indices, n_users, frac, seed=1234):
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time())) # 设置种子
    rng = random.Random(rng_seed) # <random.Random object at 0x561bda4e6f30>
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(indices) * frac)
    selected_indices = rng.sample(indices, n_samples) # 从list(range(len(dataset)))中无放回随机抽样n_samples条数据【对应的索引】,抽样后list(range(len(dataset)))保持不变
    n_per_users = int(n_samples / n_users) # 取整


    users_indices = [[] for _ in range(n_users)]
    for i in range(n_users):
        users_indices[i] = rng.sample(selected_indices, n_per_users)
        selected_indices = list(set(selected_indices) - set(users_indices[i]))
        
    return users_indices

def allocation_shards(indices, n_classes, dataset, n_users, shards, edge_frac, seed=1234):
    """
    :param train_dataset:
    :param test_dataset:
    :param shards:
    :param edge_frac:  
    :param n_users:
    :return:
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    total_shards = shards * n_users# 总shard数
    # shard_size = int(len(train_dataset.data) / total_shards) #整除 每个shard的大小
    shard_size = int(len(indices) / total_shards)

    # print("shard_size", shard_size)
    idx_shard = [i for i in range(total_shards)] # shard索引 list [0,1,...]
    dict_users = {i: np.array([],dtype=int) for i in range(n_users)} # useri对应的train数据 字典{i:np.array[],..}
    indices = np.arange(total_shards * shard_size) # 所有数据对应的索引
    labels = np.arange(n_classes) # dataset.targets # 标签label2index = {k:[] for k in range(n_classes)}()
    dict_label_dist = {i: np.array([]) for i in range(n_users)} # useri对应的标签 字典{i:np.array[],..}

    # sort labels
    labels = []
    for idx in indices:
        _, label = dataset[idx]
        labels.append(label)
    labels = np.array(labels)
    idxs_labels = np.vstack((indices, labels)) #数据和标签对应

    #array([[0, 1, 2, 3, 4],
    #   [0, 1, 0, 0, 1]])
    # 逗号前表示怎么取第一维，逗号后表示怎么取第二维 每个逗号表示每一维怎么取
    # 返回原始数据索引与标签排序后的索引的对应关系
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]#argsort返回排序索引 标签为0的在前半部分，标签为1的在后半部分
    
    indices = idxs_labels[0, :] # 原始标签索引
    label_count = np.bincount(idxs_labels[1])#标签0,1,...的数据个数 array([n0,n1,...])

    # generate adj
    A = np.zeros((n_users, n_users))
    num_label = len(set(labels))
    label_dist = [[] for _ in range(num_label)] # 每个标签对应的客户端

    # partitions for train data
    for i in range(n_users):
       
        # 返回选中的shard索引
        rand_set = np.random.choice(idx_shard, shards, replace=False) #array([idx1,idx2,...]) random choice 5 shards from total shards
        idx_shard = list(set(idx_shard) - set(rand_set))
        # print(rand_set) # [304 340  47  67 479]
        # print(shard_size) # 100
        # rand_set * shard_size [30400 34000  4700  6700 47900] 每个shard在排好序数据中的开始位置索引
        selected_labels = idxs_labels[1, rand_set * shard_size] # [6 6 0 1 9]   # shard是从label排好序的数据索引中选取的，得到标签
        label_type = np.array(list(set(selected_labels)))
        print(label_type)
        #print("label_type", label_type) # [0 1 6 9]
        sample_size = [np.count_nonzero(selected_labels == j) for j in label_type] # [1, 1, 2, 1] the number of shards of each label for client i
      
        #shard_size = int(shard_size * shards / len(label_type))
        dict_label_dist[i] = np.array((label_type, sample_size)) # {0: array([[0, 1, 6, 9],[1, 1, 2, 1]]),...) #
        # useri对应的标签 字典{i:np.array[],..}

        for j, l in enumerate(label_type): # label j assigned to client i
            # 计算user i取原始数据标签l的中哪些索引
            start_idx = sum(label_count[0:l]) # label_count #标签0,1,...的数据个数 array([n0,n1,...]) 标签l的开始索引
            end_idx = start_idx + label_count[l] # 标签l的结束索引
            sample_array = indices[start_idx: end_idx] # label l total samples 
            dict_users[i] = np.concatenate(
                (dict_users[i], np.random.choice(
                    sample_array, sample_size[j] * shard_size, replace=False)), axis=0)
                    # sample_size[j] 需要从标签l中取几个shard
                    #random choice 5 shards from total label j shards
                    # 返回取到数据的索引

        # for cifar-100, control the sparsity of A 
        # dict_users[i] [17932. 23099.  9059. 22924. 15949. 15887.  3607. 49615. 20686.  7700 15000. 12509. 33148. 36924.
        label_size = np.array([np.count_nonzero(
            labels[dict_users[i].astype(int)] == j) for j in label_type])
        # label_size [100 100 200 100]
        pram_label_idx = np.array(sorted(range(len(label_size)),
                key=lambda i: label_size[i])[min(-5, shards):]) # [0 1 3 2]
        # pram_label_idx: sorted index according to lable_size 

        for label_type in label_type[pram_label_idx]: # [0, 1, 9, 6]
            label_dist[label_type].append(i)

    # prepare A
    link_list = []
    for user_arr in label_dist: # [[client_i, client_j],...] # label_i - client_j
        for user_a in user_arr:
            for user_b in user_arr:
                link_list.append([user_a, user_b])

    # link_list.append([user_a, user_b]) [[usera,userb],...]
    link_sample = list(range(len(link_list)))
    link_idx = np.random.choice(link_sample, int(edge_frac * len(link_list)), replace=False)
    #边索引
    for idx in link_idx:
        # A[link_list[idx][0], link_list[idx][1]] = A[link_list[idx][0], link_list[idx][1]] + 1
        A[link_list[idx][0], link_list[idx][1]] = 1
        # A[3,4]
    
    users_indices = []
    for key,value in dict_users.items():
        users_indices.append(list(value))

    return users_indices, torch.tensor(A, dtype=torch.float32)

def pathological_non_iid_split(indices, n_classes, dataset, n_users, n_classes_per_user, frac, seed=1234):
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(indices) * frac)
    print(n_samples)
    selected_indices = rng.sample(indices, n_samples)

    label_dist = [[] for _ in range(num_label)] 
    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)
    #for key,item in label2index.items():
        #print(key,len(item))
    #input()

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label] #健对应的值

    n_shards = n_users * n_classes_per_user # 用户数量 * 每个用户的类数 
    shards = iid_divide(sorted_indices, n_shards) # sorted_indices分成n shards 份 [[shard1],[shard2],...]
    random.shuffle(shards) # 打乱
    user_shards = iid_divide(shards, n_users) # shards分成n_users份 [[user1],[user2]...] len = 30
    #print(len(user_shards))
    #input()

    user2label = { i:{} for i in range(n_users)}
    print(user2label)
    input()
    users_indices = [[] for _ in range(n_users)]
    for user_id in range(n_users):
        useri_label = []
        for shard in user_shards[user_id]:
            users_indices[user_id] += shard
            for i,idx in enumerate(shard):
                _, label1 = dataset[idx]
                
                useri_label.append(label1)
        unique_label = set(useri_label)
        label_cnt = { mapp[label] : 0 for label in unique_label}
        for label in unique_label:
            label_cnt[mapp[label]] = useri_label.count(label)
        user2label[user_id] = label_cnt
    for key,item in user2label.items():
        print(item)
    #print(user2label)   
    return users_indices

def dirichlet_non_iid_split(indices, n_users, dataset, n_classes, alpha, frac, n_clusters=-1, seed=1234):
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    #rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters) # [[],[],...[]] 每个簇之间的标签没有重合

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels: 
            label2cluster[label] = group_idx #标签对应的簇

    # get subset
    n_samples = int(len(indices) * frac)

    selected_indices = rng.sample(indices, n_samples)
    
    # 建立簇与数据索引之间的关系
    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        _, label = dataset[idx] # 这条数据对应的标签
        group_id = label2cluster[label]  #这条数据对应的簇号
        clusters_sizes[group_id] += 1 # 簇size++
        clusters[group_id].append(idx) # 向簇加入这条数据的索引

    for _, cluster in clusters.items():
        rng.shuffle(cluster) # 打乱每个簇里的索引号

    users_counts = np.zeros((n_clusters, n_users), dtype=np.int64)  # number of samples by client from each cluster
    #（m，i）第i个用户来自第m个簇的数量

    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_users))
        users_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)  # 对第cluster_id个簇的狄利克雷取样

    

    users_counts = np.cumsum(users_counts, axis=1)

    users_indices = [[] for _ in range(n_users)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], users_counts[cluster_id]) # 第i个簇对所有用户的划分
        
        # 将每个用户的样本累加
        for user_id, indices in enumerate(cluster_split):
            users_indices[user_id] += indices

    label_counts = np.zeros([n_users, n_classes])    
    for i, user_indice in enumerate(users_indices):
        for idx in user_indice:
            _, label = dataset[idx]
            label_counts[i][label]+=1   
    np.set_printoptions(suppress=True) 
    print(label_counts)
    
    data = pd.DataFrame(label_counts)

    writer = pd.ExcelWriter('./users_counts.xlsx')		# 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    writer.save()

    writer.close()
    return users_indices


def label_swapped_non_iid_split(indices, dataset, n_users, frac, k=4, seed=1234):
    return split_iid(indices, n_users, frac, seed=1234)

    k = 4
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(indices) * frac)
    print(n_samples)
    selected_indices = rng.sample(indices, n_samples)

    '''
    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)

    
    for idx in groups[0]:
        _, label = dataset[idx]
        if label == 1:
    '''
    groups = iid_divide(selected_indices, k) #4组
    #group 1 :swap  1 2 
    #group 2 :swap  3 4 
    #group 3 :swap  5 6 
    #group 4 :swap  7 8 

    n_user_per_cluster = int(n_users / k)
    users_indices = [[] for _ in range(n_users)]
    cur_user_id = 0
    for i in range(k):
        user_indices_cluster = iid_divide(groups[i], n_user_per_cluster) # 五组
        for user_id_in_cluster in range(n_user_per_cluster):
            users_indices[cur_user_id] = user_indices_cluster[user_id_in_cluster]
            cur_user_id += 1            
    return users_indices

