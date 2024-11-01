import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# from models.GCN import GCN, SAGENet, GraphSAGE, GAT, GCNII
from models.GCN import GCN, SAGENet, GraphSAGE, GAT, GCNII, GCN_list
from utils.setup import get_config, write_to_file
from utils.setup import test_while_training, test_while_training_in_batch, test_while_training_in_batch_idx
from utils.dataloader import load_subdata, load_ogbdata, load_ogbidx

import gc

def train(model, data, index, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    import datetime
    log_file_name = config['pre_train_log_dir'] + '{}_hidden{}_{}_epoch{}.log'.format(config['model_name'], config['hidden_size'],
                                                                                      config['dataset_name'], config['pre_train_epochs'])
    log_start_str = "================== Log on " + datetime.datetime.now().strftime('%Y-%m-%d at %H:%M:%S') + " ==================\n"

    for epoch in range(1, config['pre_train_epochs'] + 1):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out, _, _ = model(data.x, data.edge_index)  # Perform a single forward pass.

        # Compute the loss solely based on the training nodes.
        loss = criterion(out[index.old_train_mask], data.y[index.old_train_mask])

        loss.backward()   # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        val_acc, test_acc = test_while_training(model, data, index.old_val_mask, index.old_test_mask)
        # print('Epoch {:03d}, train_loss: {:.4f} | val_acc: {:.4f}, test_acc: {:.4f}'.format(epoch, loss, val_acc, test_acc))

        # Start to log and print
        if epoch % config['print_every_epochs'] == 0:
            format_str = 'Epoch {:03d}, train_loss: {:.4f} | val_acc: {:.4f}, test_acc: {:.4f}'.format(epoch, loss, val_acc, test_acc)
            print(format_str)


def test(model, data, old_test_mask):
    model.eval()
    out, _, _ = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[old_test_mask] == data.y[old_test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(old_test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


def train_in_batch(model, data, index, train_loader, subgraph_loader, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    import datetime
    log_file_name = config['pre_train_log_dir'] + '{}_hidden{}_{}_epoch{}.log'.format(
                    config['model_name'], config['hidden_size'], config['dataset_name'], config['pre_train_epochs'])
    # log_file_name = config['pre_train_log_dir'] + 'SAGE_hidden64_ogbn-arxiv_epoch200.log'   # ⭐
    log_start_str = "================== Log on " + datetime.datetime.now().strftime('%Y-%m-%d at %H:%M:%S') + " ==================\n"

    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    for epoch in range(1, config['pre_train_epochs'] + 1):
        model.train()

        total_loss = 0
        for batch_size, n_id, adjs in train_loader:
            adjs = [adj.to(device) for adj in adjs]     # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            out, _, _ = model(x[n_id], adjs)
            loss = criterion(out, y[n_id[:batch_size]])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss)

            gc.collect()
            torch.cuda.empty_cache()

        loss = total_loss / len(train_loader)

        # Start to log and print
        if epoch % config['print_every_epochs'] == 0:
            if config['dataset_name'] == 'ogbn-arxiv':
                evaluator = Evaluator(name='ogbn-arxiv')
                train_acc, val_acc, test_acc = test_while_training_in_batch_idx(model, x, y, subgraph_loader, evaluator, index)
            else:
                train_acc, val_acc, test_acc = test_while_training_in_batch(model, index, x, y, subgraph_loader)

            format_str = 'Epoch {:03d}, train_loss: {:.4f} | train_acc: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}'.format(
                epoch, loss, train_acc, val_acc, test_acc)  # ⭐
            print(format_str)


def main(args):
    config = get_config(args.config)

    # Prepare datasets
    if config['dataset_name'] == 'Cora':
        dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    elif config['dataset_name'] == 'CiteSeer':
        dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
    elif config['dataset_name'] == 'PubMed':
        dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
    elif config['dataset_name'] == 'ogbn-arxiv':
        # dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/OGBNode/', transform=T.ToSparseTensor())
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/OGBNode/', transform=NormalizeFeatures())

    num_old = config['num_old']
    num_new = dataset.num_classes - num_old
    # num_new = 2

    # Dataloader creation
    data = dataset[0]

    print("================== Getting subdata split idx...... ==================")
    index = type('IndexObject', (object,), {})()

    if config['dataset_name'] == 'ogbn-arxiv':
        data.y = data.y.T[0]
        index.old_train_mask, index.old_test_mask, index.old_val_mask = load_ogbdata(data, num_old=num_old, split="old", split_idx=dataset.get_idx_split())
    else:
        index.old_train_mask, index.old_test_mask, index.old_val_mask = load_subdata(data, num_old=num_old, split="old")
        # new_train_mask, new_test_mask, _ = load_subdata(data, num_old=num_old, split="new")
        # all_train_mask, all_test_mask, _ = load_subdata(data, num_old=num_old, split="old+new")

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    data = data.to(device)

    if config['model_name'] == 'GCN':
        model = GCN(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GraphSAGE':
        model = GraphSAGE(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GAT':
        model = GAT(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GCNII':
        model = GCNII(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)

    elif config['model_name'] == 'GCN_list':
        model = GCN_list(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)

    elif config['model_name'] == 'cluster-GCN':
        model = GCN(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
        data.y = data.y.T[0]
        cluster_data = ClusterData(data, num_parts=32)  # 1. Create subgraphs.
        train_loader = ClusterLoader(cluster_data, batch_size=64, shuffle=True)  # 2. Stochastic partioning scheme.

        print("================== Start pre-training in batches...... ==================")
        # train_in_batch(model, data, index, train_loader, subgraph_loader, config, device)
        # train(model, data, index, config)

        model_dir = config['model_dir'] + \
                    'warmup_{}_hidden{}_{}_epoch{}.pth'.format(config['model_name'], config['hidden_size'],
                                                        config['dataset_name'], config['pre_train_epochs'])
        torch.save(model.state_dict(), model_dir)
        print("\nwarmup model saved to {}.".format(model_dir))
        return

    elif config['model_name'] == 'SAGE':
        if config['dataset_name'] == 'Cora':
            model = SAGENet(dataset=dataset, hidden=config['hidden_size'], num_layers=3, num_old=num_old, num_new=num_new).to(device)

            train_loader = NeighborSampler(data.edge_index, node_idx=index.old_train_mask,
                                           sizes=[10, 10], batch_size=1024, shuffle=True)
            subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                              sizes=[-1], batch_size=1024, shuffle=False)

        elif config['dataset_name'] == 'ogbn-arxiv':
            model = SAGENet(dataset=dataset, hidden=64, num_layers=3, num_old=num_old, num_new=num_new).to(device)

            train_loader = NeighborSampler(data.adj_t, node_idx=index.old_train_idx,
                                           sizes=[15, 10, 5], batch_size=4096, shuffle=True)
            subgraph_loader = NeighborSampler(data.adj_t, node_idx=None,
                                              sizes=[-1],batch_size=4096, shuffle=False)

        print("================== Start pre-training in batches...... ==================")
        train_in_batch(model, data, index, train_loader, subgraph_loader, config, device)

        model_dir = config['model_dir'] + \
                    'warmup_{}_hidden{}_{}_epoch{}.pth'.format(config['model_name'], config['hidden_size'],
                                                        config['dataset_name'], config['pre_train_epochs'])
        torch.save(model.state_dict(), model_dir)
        print("\nwarmup model saved to {}.".format(model_dir))
        return


    # train the model
    model.to(device)
    print("================== Start pre-training...... ==================")
    train(model, data, index, config)
    test_acc = test(model, data, index.old_test_mask)

    print("="*20)
    print(f'Test Accuracy: {test_acc:.4f}')


    # save the warmed-up model
    import os
    model_dir = config['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_name = 'warmup_{}_hidden{}_{}_epoch{}.pth'.format(config['model_name'], config['hidden_size'],
                                                            config['dataset_name'], config['pre_train_epochs'])
    model_dir = model_dir + model_name

    torch.save(model.state_dict(), model_dir)
    print("\nwarmup model saved to {}.".format(model_dir))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='path to the config file')
    args = parser.parse_args()
    main(args)
