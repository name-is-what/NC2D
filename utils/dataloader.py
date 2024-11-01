import torch
import numpy as np


def load_subdata(data, num_old, split):
    sub_train_mask = torch.tensor([False] * data.num_nodes)
    sub_test_mask = torch.tensor([False] * data.num_nodes)
    sub_val_mask = torch.tensor([False] * data.num_nodes)

    if split=='old':
        for i in range(data.num_nodes):
            if data.y[i] in range(num_old) and data.train_mask[i] == True: sub_train_mask[i] = True
            if data.y[i] in range(num_old) and data.test_mask[i] == True: sub_test_mask[i] = True
            if data.y[i] in range(num_old) and data.val_mask[i] == True: sub_val_mask[i] = True

    elif split=='new':
        for i in range(data.num_nodes):
            if data.y[i] not in range(num_old) and data.train_mask[i] == True: sub_train_mask[i] = True
            if data.y[i] not in range(num_old) and data.test_mask[i] == True: sub_test_mask[i] = True
            if data.y[i] not in range(num_old) and data.val_mask[i] == True: sub_val_mask[i] = True

    elif split=='old+new':
        # for i in range(data.num_nodes):
        #     if data.y[i] in range(num_old): data.train_mask[i] = True
        #     else: data.train_mask[i] = False
        sub_train_mask, sub_test_mask, sub_val_mask = data.train_mask, data.test_mask, data.val_mask

    return sub_train_mask, sub_test_mask, sub_val_mask


def load_ogbdata(data, num_old, split, split_idx):
    sub_train_mask = torch.tensor([False] * data.num_nodes)
    sub_test_mask = torch.tensor([False] * data.num_nodes)
    sub_val_mask = torch.tensor([False] * data.num_nodes)

    if split=='old':
        for i in split_idx['train']:
            if data.y[i] in range(num_old): sub_train_mask[i] = True
        for i in split_idx['test']:
            if data.y[i] in range(num_old): sub_test_mask[i] = True
        for i in split_idx['valid']:
            if data.y[i] in range(num_old): sub_val_mask[i] = True

    elif split=='new':
        for i in split_idx['train']:
            if data.y[i] not in range(num_old): sub_train_mask[i] = True
        for i in split_idx['test']:
            if data.y[i] not in range(num_old): sub_test_mask[i] = True
        for i in split_idx['valid']:
            if data.y[i] not in range(num_old): sub_val_mask[i] = True

    elif split=='old+new':
        sub_train_mask[split_idx['train']] = True
        sub_test_mask[split_idx['test']] = True
        sub_val_mask[split_idx['valid']] = True

    # elif split=='new':
    #     for i in split_idx['train']:
    #         if data.y[i] in range(num_old,num_old+2): sub_train_mask[i] = True
    #     for i in split_idx['test']:
    #         if data.y[i] in range(num_old,num_old+2): sub_test_mask[i] = True
    #     for i in split_idx['valid']:
    #         if data.y[i] in range(num_old,num_old+2): sub_val_mask[i] = True
    # elif split=='old+new':
    #     for i in split_idx['train']:
    #         if data.y[i] in range(num_old+2): sub_train_mask[i] = True
    #     for i in split_idx['test']:
    #         if data.y[i] in range(num_old+2): sub_test_mask[i] = True
    #     for i in split_idx['valid']:
    #         if data.y[i] in range(num_old+2): sub_val_mask[i] = True

    return sub_train_mask, sub_test_mask, sub_val_mask


def load_ogbidx(dataset, num_old=25, split='train'):
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    old_idx, new_idx = np.empty(shape=(1,0)), np.empty(shape=(1,0))
    # train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    for i in split_idx[split]:
        if data.y[i] in range(num_old): old_idx = np.append(old_idx, i)
        else: new_idx = np.append(new_idx, i)

    old_idx, new_idx = torch.tensor(old_idx, dtype=torch.int64), torch.tensor(new_idx, dtype=torch.int64)
    return old_idx, new_idx


"""
from torch_geometric.loader import ClusterData, ClusterLoader

cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)  # 2. Stochastic partioning scheme.
"""