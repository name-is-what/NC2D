import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.transforms as T

from models.GCN import GCN, GraphSAGE, GAT, GCNII
from utils.setup import get_config, write_to_file
from utils.setup import test_while_training
from utils.dataloader import load_subdata, load_ogbdata
from utils.calculator import get_loss_bce, get_loss_ce, get_loss_mse, get_loss_replay, get_loss_kd
from utils.calculator import cal_mean_sig

import gc

##### python incd_expt_aug.py --config config/gcn_cora.yml

def incd_train(model, old_model, data, index, config, device, old_mean, old_sig):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    import datetime
    log_file_name = config['ncd_train_log_dir'] + '{}_hidden{}_{}_epoch{}.log'.format(config['model_name'], config['hidden_size'],
                                                                                      config['dataset_name'], config['pre_train_epochs'])
    log_start_str = "================== Log on " + datetime.datetime.now().strftime('%Y-%m-%d at %H:%M:%S') + " ==================\n"
    # write_to_file(log_start_str, log_file_name)   # ⭐

    #
    # import pandas as pd
    # df = pd.DataFrame(columns = ["Loss"]) # columns列名
    # df.index.name = "Epoch"

    for epoch in range(1, config['ncd_train_epochs'] + 1):
        model.train()
        optimizer.zero_grad()  # Clear gradients.

        out1, out2, feat = model(data.x, data.edge_index)  # Perform a single forward pass.
        # out1, out2, feat = model(data.x, data.x, data.edge_index)     ## GCNII

        # vice_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)
        # eta=1.0, eta=0.1
        # vice_data = data + 0.1*torch.normal(0,torch.ones_like(data)*data.std()).to(device)
        # vice_out1, vice_out2, _ = model(vice_data.x, vice_data.edge_index)
        # vice_out1 = out1 + 0.1*torch.normal(0,torch.ones_like(out1)*out1.std()).to(device)

        vice_out2 = out2 + 0.2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)
        # vice_out2 = out2 + 2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)

        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)

        # loss_novel  = get_loss_bce(prob2, prob2_bar, feat[index.new_train_mask], device) + \
        #               get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],
        #                           num_old=config['num_old'], rampup_length=config['rampup_length'],
        #                           increment_coefficient=config['increment_coefficient']) + \
        #               get_loss_mse(prob2, prob2_bar, config, epoch)

        ## arxiv
        criterion = torch.nn.CrossEntropyLoss()
        loss_novel  = get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],
                                  num_old=config['num_old'], rampup_length=config['rampup_length'],
                                  increment_coefficient=config['increment_coefficient']) + \
                      get_loss_mse(prob2, prob2_bar, config, epoch) + \
                    criterion(out1[index.new_train_mask], data.y[index.new_train_mask])
        #             # criterion(out1[index.all_train_mask], data.y[index.all_train_mask])



        loss_past = get_loss_replay(model, old_mean, old_sig,
                                    num_old=config['num_old'], lambda_proto=config['lambda_proto'], device=device) + \
                    get_loss_kd(old_model, feat, data, w_kd=config['w_kd'])
        # loss_past = get_loss_kd(old_model, feat, data, w_kd=config['w_kd'])
        # 打印当前进程的GPU占用总量
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # loss = loss_novel + loss_past
        # loss = loss_novel * 1.2 + loss_past*0.5
        loss = loss_novel * 1.8 + loss_past*0.5     ## review

        # df.loc[epoch] = loss.item()

        loss.backward()   # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        val_acc, test_acc = test_while_training(model, data, index.new_val_mask, index.new_test_mask)
        old_val_acc, old_test_acc = test_while_training(model, data, index.old_val_mask, index.old_test_mask)
        all_val_acc, all_test_acc = test_while_training(model, data, index.all_test_mask, index.all_test_mask)
        # all_val_acc, all_test_acc = test_while_training(model, data, data.val_mask, data.test_mask)

        # Start to log and print
        if epoch % config['print_every_epochs'] == 0:
            # format_str = 'Epoch {:03d}, train_loss: {:.4f} | val_acc: {:.4f}, test_acc: {:.4f}'.format(epoch, loss, val_acc, test_acc)
            # format_str = 'Epoch {:03d}, train_loss: {:.4f} | new_val_acc: {:.4f}, new_test_acc: {:.4f}, old_val_acc: {:.4f}, old_test_acc: {:.4f}, all_test_acc: {:.4f}'.format(epoch, loss, val_acc, test_acc, old_val_acc, old_test_acc, all_test_acc)
            format_str = 'Epoch {:03d}, train_loss: {:.4f} | new_val_acc: {:.4f}, new_test_acc: {:.4f}, old_val_acc: {:.4f}, old_test_acc: {:.4f}, all_val_acc: {:.4f}, all_test_acc: {:.4f}'.format(epoch, loss, val_acc, test_acc, old_val_acc, old_test_acc, all_val_acc, all_test_acc)
            # write_to_file(format_str + '\n', log_file_name)   # ⭐
            print(format_str)
    # df.plot()
    # import matplotlib.pyplot as plt
    # plt.title('1:1, 600 epoch, eta=1, 1:8', loc='center')
    # plt.show()

def incd_train_re(model, old_model, data, index, config, device, old_mean, old_sig):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    for epoch in range(1, config['ncd_train_epochs'] + 1):
        model.train()
        optimizer.zero_grad()  # Clear gradients.

        out1, out2, feat = model(data.x, data.edge_index)  # Perform a single forward pass.


        vice_out2 = out2 + 0.2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)
        # vice_out2 = out2 + 2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)

        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)

        loss_novel  = get_loss_bce(prob2, prob2_bar, feat[index.new_train_mask], device) + \
                      get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],
                                  num_old=config['num_old'], rampup_length=config['rampup_length'],
                                  increment_coefficient=config['increment_coefficient']) + \
                      get_loss_mse(prob2, prob2_bar, config, epoch)

        ## arxiv
        # loss_novel  = get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],
        #                           num_old=config['num_old'], rampup_length=config['rampup_length'],
        #                           increment_coefficient=config['increment_coefficient']) + \
        #               get_loss_mse(prob2, prob2_bar, config, epoch)

        # loss_novel  = get_loss_bce(prob2[:1000], prob2_bar[:1000], feat[index.new_train_mask][:1000], device) + \
        #               get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],
        #                           num_old=config['num_old'], rampup_length=config['rampup_length'],
        #                           increment_coefficient=config['increment_coefficient']) + \
        #               get_loss_mse(prob2, prob2_bar, config, epoch)



        loss_past = get_loss_replay(model, old_mean, old_sig,
                                    num_old=config['num_old'], lambda_proto=config['lambda_proto'], device=device) + \
                    get_loss_kd(old_model, feat, data, w_kd=config['w_kd'])

        loss = loss_novel * 1.2 + loss_past*0.5

        loss.backward()   # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        val_acc, test_acc = test_while_training(model, data, index.new_val_mask, index.new_test_mask)
        old_val_acc, old_test_acc = test_while_training(model, data, index.old_val_mask, index.old_test_mask)
        all_val_acc, all_test_acc = test_while_training(model, data, index.all_test_mask, index.all_test_mask)

        # Start to log and print
        if epoch % config['print_every_epochs'] == 0:
            format_str = 'Epoch {:03d}, train_loss: {:.4f} | new_val_acc: {:.4f}, new_test_acc: {:.4f}, old_val_acc: {:.4f}, old_test_acc: {:.4f}, all_val_acc: {:.4f}, all_test_acc: {:.4f}'.format(epoch, loss, val_acc, test_acc, old_val_acc, old_test_acc, all_val_acc, all_test_acc)
            print(format_str)

def incd_test(model, data, split, sub_mask, num_old=4):
    model.eval()
    out1, out2, _ = model(data.x, data.edge_index)

    if split=="head1":
        pred = out1.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[sub_mask] == (data.y[sub_mask])  # Check against ground-truth labels.

    elif split=="head2":
        pred = out2.argmax(dim=1) + num_old
        test_correct = pred[sub_mask] == (data.y[sub_mask])

    test_acc = int(test_correct.sum()) / int(sub_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


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
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/OGBNode/', transform=NormalizeFeatures())

    num_old = config['num_old']
    num_new = dataset.num_classes - num_old
    # num_new = 2

    # Dataloader creation
    data = dataset[0]

    print("================== Getting subdata split idx...... ==================")
    index = type('IndexObject', (object,), {})()
    # index.old_train_mask, index.old_test_mask, index.old_val_mask = load_subdata(data, num_old=num_old, split="old")
    # index.new_train_mask, index.new_test_mask, index.new_val_mask = load_subdata(data, num_old=num_old, split="new")
    # index.all_train_mask, index.all_test_mask, index.all_val_mask = load_subdata(data, num_old=num_old, split="old+new")
    if config['dataset_name'] == 'ogbn-arxiv':
        data.y = data.y.T[0]

        import os
        import pickle

        # 定义保存和加载mask的文件名
        masks_save_file = 'data/arxiv-masks.pkl'

        # 检查文件是否存在
        if os.path.exists(masks_save_file):
            with open(masks_save_file, 'rb') as f:
                index.old_train_mask, index.old_test_mask, index.old_val_mask, index.new_train_mask, index.new_test_mask, index.new_val_mask, index.all_train_mask, index.all_test_mask, index.all_val_mask = pickle.load(f)
                print("Loaded masks from file:", masks_save_file)
        else:
            index.old_train_mask, index.old_test_mask, index.old_val_mask = load_ogbdata(data, num_old=num_old, split="old", split_idx=dataset.get_idx_split())
            index.new_train_mask, index.new_test_mask, index.new_val_mask = load_ogbdata(data, num_old=num_old, split="new", split_idx=dataset.get_idx_split())
            index.all_train_mask, index.all_test_mask, index.all_val_mask = load_ogbdata(data, num_old=num_old, split="old+new", split_idx=dataset.get_idx_split())
            # 然后将masks保存到文件
            with open(masks_save_file, 'wb') as f:
                pickle.dump((index.old_train_mask, index.old_test_mask, index.old_val_mask, index.new_train_mask, index.new_test_mask, index.new_val_mask, index.all_train_mask, index.all_test_mask, index.all_val_mask), f)
                print("Saved masks to file:", masks_save_file)

        # index.old_train_mask, index.old_test_mask, index.old_val_mask = load_ogbdata(data, num_old=num_old, split="old", split_idx=dataset.get_idx_split())
        # index.new_train_mask, index.new_test_mask, index.new_val_mask = load_ogbdata(data, num_old=num_old, split="new", split_idx=dataset.get_idx_split())
        # index.all_train_mask, index.all_test_mask, index.all_val_mask = load_ogbdata(data, num_old=num_old, split="old+new", split_idx=dataset.get_idx_split())
    else:
        index.old_train_mask, index.old_test_mask, index.old_val_mask = load_subdata(data, num_old=num_old, split="old")
        index.new_train_mask, index.new_test_mask, index.new_val_mask = load_subdata(data, num_old=num_old, split="new")
        index.all_train_mask, index.all_test_mask, index.all_val_mask = load_subdata(data, num_old=num_old, split="old+new")

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    data = data.to(device)

    # model = GCN(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    if config['model_name'] == 'GCN':
        model = GCN(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GraphSAGE':
        model = GraphSAGE(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GAT':
        model = GAT(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GCNII':
        model = GCNII(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)

    # get model from warmup
    print("================== Load warmup model...... ==================")
    warmup_model_dir = config['model_dir'] + \
                'warmup_{}_hidden{}_{}_epoch{}.pth'.format(config['model_name'], config['hidden_size'],
                                                    config['dataset_name'], config['pre_train_epochs'])
    # warmup_model_dir = config['model_dir'] + 'warmup_SAGE_hidden64_ogbn-arxiv_epoch200.pth'
    state_dict = torch.load(warmup_model_dir)
    model.load_state_dict(state_dict, strict=False)

    import copy
    old_model = copy.deepcopy(model).to(device)
    old_model.eval()

    save_weight = model.head1.weight.data.clone()   # save the weights of head-1
    save_bias = model.head1.bias.data.clone()       # save the bias of head-1
    model.head1 = nn.Linear(config['hidden_size'], num_old+num_new).to(device)       # replace the labeled-class only head-1 with the head-1-new include nodes for novel calsses
    # if config['model_name'] == 'GCNII': model.head1 = nn.Linear(data.num_features, num_old+num_new).to(device)

    model.head1.weight.data[:num_old] = save_weight       # put the old weights into the old part
    model.head1.bias.data[:] = torch.min(save_bias) - 1.    # put the bias
    model.head1.bias.data[:num_old] = save_bias

    model.l2_classifier = True

    # training
    old_mean, old_sig = cal_mean_sig(model, data, index.old_train_mask, config['hidden_size'], num_old, device)
    # print(old_mean.shape, old_sig.shape)
    print("================== Start incd-training...... ==================")
    incd_train(model, old_model, data, index, config, device, old_mean, old_sig)

    # if config['dataset_name'] == 'ogbn-arxiv':
    #     incd_train_axriv(model, old_model, data, index, config, device, old_mean, old_sig)
    # else:
    #     incd_train(model, old_model, data, index, config, device, old_mean, old_sig)

    # final testing with GCN
    print("\n================== Final Test Accuracy ==================")
    test_new_acc = incd_test(model, data, split="head1", sub_mask=index.new_test_mask)
    test_new_acc_2 = incd_test(model, data, split="head2", sub_mask=index.new_test_mask, num_old=num_old)
    test_old_acc = incd_test(model, data, split="head1", sub_mask=index.old_test_mask)
    test_all_acc = incd_test(model, data, split="head1", sub_mask=index.all_test_mask)
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')
    print(f'head2: new = {test_new_acc_2:.4f}')


    # saving incd model
    model_name = 'incd_{}_hidden{}_{}_epoch{}.pth'.format(config['model_name'], config['hidden_size'],
                                                            config['dataset_name'], config['ncd_train_epochs'])
    model_dir = config['model_dir'] + model_name

    # torch.save(model.state_dict(), model_dir)     # ⭐
    print("\nincd model saved to {}.".format(model_dir))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='path to the config file')
    args = parser.parse_args()

    main(args)