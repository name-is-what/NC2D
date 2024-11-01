import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.transforms as T

from models.GCN import GCN, GraphSAGE, GAT, GCNII
from utils.setup import get_config, test_while_training
from utils.dataloader import load_subdata
from utils.calculator import get_loss_bce, get_loss_ce, get_loss_mse, get_loss_replay, get_loss_kd
from utils.calculator import cal_mean_sig

import gc


def incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # for epoch in range(1, config['ncd_train_epochs'] + 1):
    for epoch in range(1, 600 + 1):
        model.train()
        optimizer.zero_grad()  # Clear gradients.

        out1, out2, feat = model(data.x, data.edge_index)  # Perform a single forward pass.
        vice_out2 = out2 + 0.2 * torch.normal(0, torch.ones_like(out2) * out2.std()).to(device)
        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)

        loss_bce = get_loss_bce(prob2, prob2_bar, feat[index.new_train_mask], device)
        loss_self = get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],num_old=config['num_old'],
                                rampup_length=config['rampup_length'],increment_coefficient=config['increment_coefficient'])
        loss_mse = get_loss_mse(prob2, prob2_bar, config, epoch)
        loss_replay = get_loss_replay(model, old_mean, old_sig, num_old=config['num_old'], lambda_proto=config['lambda_proto'], device=device)
        loss_kd = get_loss_kd(old_model, feat, data, w_kd=config['w_kd'])

        loss = get_loss_ablation(loss_bce, loss_self, loss_mse, loss_replay, loss_kd, abla)

        loss.backward()   # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        val_acc, test_acc = test_while_training(model, data, index.new_val_mask, index.new_test_mask)
        old_val_acc, old_test_acc = test_while_training(model, data, index.old_val_mask, index.old_test_mask)
        all_val_acc, all_test_acc = test_while_training(model, data, data.val_mask, data.test_mask)

        # Start to log and print
        if epoch % config['print_every_epochs'] == 0:
            format_str = 'Epoch {:03d}, train_loss: {:.4f} | new_val_acc: {:.4f}, new_test_acc: {:.4f}, old_val_acc: {:.4f}, old_test_acc: {:.4f}, all_val_acc: {:.4f}, all_test_acc: {:.4f}'.format(epoch, loss, val_acc, test_acc, old_val_acc, old_test_acc, all_val_acc, all_test_acc)
            print(format_str)

    print("\n=============== Ablation Study {} for Dataset {} Final Test Accuracy ===============".format(abla, config['dataset_name']))
    test_new_acc = incd_test(model, data, split="head1", sub_mask=index.new_test_mask)
    test_new_acc_2 = incd_test(model, data, split="head2", sub_mask=index.new_test_mask, num_old=config['num_old'])
    test_old_acc = incd_test(model, data, split="head1", sub_mask=index.old_test_mask)
    test_all_acc = incd_test(model, data, split="head1", sub_mask=index.all_test_mask)
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')
    print(f'head2: new = {test_new_acc_2:.4f}')
    model_dir = config['model_dir'] + abla + "_{}_hidden{}_{}_epoch{}.pth".format(config['model_name'], config['hidden_size'],config['dataset_name'], config['ncd_train_epochs'])
    torch.save(model.state_dict(), model_dir)

def incd_train_head(model, old_model, data, index, config, device, old_mean, old_sig, num_new, woST=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    for epoch in range(1, config['ncd_train_epochs'] + 1):
        model.train()
        optimizer.zero_grad()  # Clear gradients.

        out1, _, feat = model(data.x, data.edge_index)  # Perform a single forward pass.
        out2 = out1[:, -num_new:]

        vice_out2 = out2 + 0.2 * torch.normal(0, torch.ones_like(out2) * out2.std()).to(device)
        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)

        loss_bce = get_loss_bce(prob2, prob2_bar, feat[index.new_train_mask], device)
        loss_self = get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],num_old=config['num_old'],
                                rampup_length=config['rampup_length'],increment_coefficient=config['increment_coefficient'])
        loss_mse = get_loss_mse(prob2, prob2_bar, config, epoch)
        loss_replay = get_loss_replay(model, old_mean, old_sig, num_old=config['num_old'], lambda_proto=config['lambda_proto'], device=device)
        loss_kd = get_loss_kd(old_model, feat, data, w_kd=config['w_kd'])

        if woST:
            loss = get_loss_ablation(loss_bce, loss_self, loss_mse, loss_replay, loss_kd, abla="wo_ST")
        else:
            loss = loss_self * 1.2 + get_loss_ablation(loss_bce, loss_self, loss_mse, loss_replay, loss_kd, abla="wo_ST")

        loss.backward()   # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    print("\n=============== Ablation woST:{} for Dataset {} Final Test Accuracy ===============".format(woST, config['dataset_name']))
    test_new_acc = incd_test(model, data, split="head1", sub_mask=index.new_test_mask)
    test_new_acc_2 = incd_test(model, data, split="head2", sub_mask=index.new_test_mask, num_old=config['num_old'])
    test_old_acc = incd_test(model, data, split="head1", sub_mask=index.old_test_mask)
    test_all_acc = incd_test(model, data, split="head1", sub_mask=index.all_test_mask)
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')
    print(f'head2: new = {test_new_acc_2:.4f}')

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


def get_loss_ablation(loss_bce, loss_self, loss_mse, loss_replay, loss_kd, abla):
    if abla == "wo_FD_FR":
        loss_novel = loss_bce + loss_self + loss_mse
        loss = loss_novel * 1.2

    elif abla == "wo_FD":
        loss_novel = loss_bce + loss_self + loss_mse
        loss_past = loss_replay
        loss = loss_novel * 1.2 + loss_past * 0.5

    elif abla == "wo_FR":
        loss_novel = loss_bce + loss_self + loss_mse
        loss_past = loss_kd
        loss = loss_novel * 1.2 + loss_past * 0.5

    elif abla == "wo_ST":
        loss_novel = loss_bce + loss_mse
        loss_past = loss_replay + loss_kd
        loss = loss_novel * 1.2 + loss_past * 0.5

    elif abla == "wo_ST_FR":
        loss_novel = loss_bce + loss_mse
        loss_past = loss_kd
        loss = loss_novel * 1.2 + loss_past * 0.5

    elif abla == "wo_ST_FD":
        loss_novel = loss_bce + loss_mse
        loss_past = loss_replay
        loss = loss_novel * 1.2 + loss_past * 0.5

    elif abla == "wo_ST_FD_FR":
        loss_novel = loss_bce + loss_mse
        loss = loss_novel * 1.2

    elif abla == "wo_PSEUDO":
        loss_novel = loss_self + loss_mse
        loss_past = loss_replay + loss_kd
        loss = loss_novel * 1.2 + loss_past * 0.5
    elif abla == "wo_PERTURB":
        loss_novel = loss_bce + loss_self
        loss_past = loss_replay + loss_kd
        loss = loss_novel * 1.2 + loss_past * 0.5

    return loss


def get_new_model(data, config, num_old, num_new, device):
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
    print("=============== Load warmup model...... ===============")
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

    model.head1.weight.data[:num_old] = save_weight       # put the old weights into the old part
    model.head1.bias.data[:] = torch.min(save_bias) - 1.    # put the bias
    model.head1.bias.data[:num_old] = save_bias

    model.l2_classifier = True

    return model, old_model


def main(args):
    config = get_config(args.config)

    # Prepare datasets
    if config['dataset_name'] == 'Cora':
        dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    elif config['dataset_name'] == 'CiteSeer':
        dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
    elif config['dataset_name'] == 'PubMed':
        dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

    num_old = config['num_old']
    num_new = dataset.num_classes - num_old

    # Dataloader creation
    data = dataset[0]

    print("=============== Getting subdata split idx...... ===============")
    index = type('IndexObject', (object,), {})()
    index.old_train_mask, index.old_test_mask, index.old_val_mask = load_subdata(data, num_old=num_old, split="old")
    index.new_train_mask, index.new_test_mask, index.new_val_mask = load_subdata(data, num_old=num_old, split="new")
    index.all_train_mask, index.all_test_mask, index.all_val_mask = load_subdata(data, num_old=num_old, split="old+new")

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    data = data.to(device)

    # model = GCN(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)

    # # get model from warmup
    # print("=============== Load warmup model...... ===============")
    # warmup_model_dir = config['model_dir'] + \
    #             'warmup_{}_hidden{}_{}_epoch{}.pth'.format(config['model_name'], config['hidden_size'],
    #                                                 config['dataset_name'], config['pre_train_epochs'])
    # # warmup_model_dir = config['model_dir'] + 'warmup_SAGE_hidden64_ogbn-arxiv_epoch200.pth'
    # state_dict = torch.load(warmup_model_dir)
    # model.load_state_dict(state_dict, strict=False)

    # import copy
    # old_model = copy.deepcopy(model).to(device)
    # old_model.eval()

    # save_weight = model.head1.weight.data.clone()   # save the weights of head-1
    # save_bias = model.head1.bias.data.clone()       # save the bias of head-1
    # model.head1 = nn.Linear(config['hidden_size'], num_old+num_new).to(device)       # replace the labeled-class only head-1 with the head-1-new include nodes for novel calsses

    # model.head1.weight.data[:num_old] = save_weight       # put the old weights into the old part
    # model.head1.bias.data[:] = torch.min(save_bias) - 1.    # put the bias
    # model.head1.bias.data[:num_old] = save_bias

    # model.l2_classifier = True

    model, old_model = get_new_model(data, config, num_old, num_new, device)
    old_mean, old_sig = cal_mean_sig(model, data, index.old_train_mask, config['hidden_size'], num_old, device)

    print("=============== Start incd-training in ablation for Dataset {} ...... ===============".format(config['dataset_name']))

    # incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_FD")
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_FR")
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_FD_FR")
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_ST")
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_ST_FR")
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_ST_FD")
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_ST_FD_FR")

    model, old_model = get_new_model(data, config, num_old, num_new, device)
    incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_PSEUDO")
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_ablation(model, old_model, data, index, config, device, old_mean, old_sig, abla="wo_PERTURB")


    ## head-1 / head-2
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_head(model, old_model, data, index, config, device, old_mean, old_sig, num_new, woST=True)
    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # incd_train_head(model, old_model, data, index, config, device, old_mean, old_sig, num_new, woST=False)

    print("\nDone!")


# python incd_expt_abla.py --config config/gcn_cora.yml
# python incd_expt_abla.py --config config/gcn_citeseer.yml
# python incd_expt_abla.py --config config/gcn_pubmed2.yml
# python wiki_abla.py

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--abla', type=str, default='wo_FD')
    args = parser.parse_args()

    main(args)
