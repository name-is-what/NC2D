import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.transforms as T

import math
from torch.optim import lr_scheduler
from torch.optim import SGD
from utils.ramps import sigmoid_rampup
from utils.calculator import BCE
from utils.calculator import PairEnum

from models.GCN import GCN, Res_GCN, GraphSAGE, GAT, GCNII
from utils.setup import get_config, write_to_file
from utils.dataloader import load_subdata, load_ogbdata
from utils.calculator import get_loss_bce, get_loss_ce, get_loss_mse, get_loss_replay, get_loss_kd
from utils.calculator import cal_mean_sig
from incd_expt_abla import get_new_model

import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def incd_train_SGD(model, old_model, data, index, config, device, old_mean, old_sig):
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.1, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=170, gamma=0.1)

    # for epoch in range(1, config['ncd_train_epochs'] + 1):
    for epoch in range(200):
        model.train()
        exp_lr_scheduler.step()
        optimizer.zero_grad()  # Clear gradients.

        out1, out2, feat = model(data.x, data.edge_index)  # Perform a single forward pass.
        vice_out2 = out2 + 0.2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)
        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)

        loss_novel  = get_loss_bce(prob2, prob2_bar, feat[index.new_train_mask], device) + \
                      get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],
                                  num_old=config['num_old'], rampup_length=config['rampup_length'],
                                  increment_coefficient=config['increment_coefficient']) + \
                      get_loss_mse(prob2, prob2_bar, config, epoch)
        loss_past = get_loss_replay(model, old_mean, old_sig,
                                    num_old=config['num_old'], lambda_proto=config['lambda_proto'], device=device) + \
                    get_loss_kd(old_model, feat, data, w_kd=config['w_kd'])
        # loss = loss_novel + loss_past
        loss = loss_novel * 1.2 + loss_past*0.5

        loss.backward()   # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    print("\n=============== Study OG for Dataset {} Final Test Accuracy ===============".format(config['dataset_name']))
    test_new_acc = incd_test(model, data, split="head1", sub_mask=index.new_test_mask)
    test_new_acc_2 = incd_test(model, data, split="head2", sub_mask=index.new_test_mask, num_old=config['num_old'])
    test_old_acc = incd_test(model, data, split="head1", sub_mask=index.old_test_mask)
    test_all_acc = incd_test(model, data, split="head1", sub_mask=index.all_test_mask)
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')
    print(f'head2: new = {test_new_acc_2:.4f}')


# AutoNovel-2020-ICLR baseline on GCN
def AutoNovel_train(data, index, config, num_old, num_new, device):
    # init
    model, old_model = get_new_model(data, config, num_old, num_new, device)
    model.l2_classifier = False
    frozen_layers = ['head1']
    freeze_layers(model, frozen_layers, True)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.1, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=170, gamma=0.1)
    for epoch in range(200):
        model.train()
        exp_lr_scheduler.step()
        out1, out2, feat = model(data.x, data.edge_index)  # Perform a single forward pass.

        vice_out1 = out1 + 0.2*torch.normal(0,torch.ones_like(out1)*out1.std()).to(device)
        prob1, prob1_bar = F.softmax(out1[index.new_train_mask], dim=1), F.softmax(vice_out1[index.new_train_mask], dim=1)

        vice_out2 = out2 + 0.2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)
        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)
        loss = get_loss_bce(prob2, prob2_bar, feat[index.new_train_mask], device) + \
               get_loss_mse(prob1, prob1_bar, config, epoch) + get_loss_mse(prob2, prob2_bar, config, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n=============== AutoNovel for Dataset {} Final Test Accuracy ===============".format(config['dataset_name']))
    test_new_acc = incd_test(model, data, split="head1", sub_mask=index.new_test_mask)
    test_new_acc_2 = incd_test(model, data, split="head2", sub_mask=index.new_test_mask, num_old=config['num_old'])
    test_old_acc = incd_test(model, data, split="head1", sub_mask=index.old_test_mask)
    test_all_acc = incd_test(model, data, split="head1", sub_mask=index.all_test_mask)
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')
    print(f'head2: new = {test_new_acc_2:.4f}')

    model_dir = config['model_dir'] + "AutoNovel_{}_hidden{}_{}_epoch{}.pth".format(config['model_name'], config['hidden_size'],config['dataset_name'], config['ncd_train_epochs'])
    # torch.save(model.state_dict(), model_dir)


# NCL-2021-CVPR baseline on GCN
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class NCLMemory(nn.Module):
    """Memory Module for NCL"""
    def __init__(self, inputSize, K=2000, T=0.05, num_class=5, knn=None, w_pos=0.2, hard_iter=5, num_hard=400, hard_negative_start=1000):
        super(NCLMemory, self).__init__()
        self.inputSize = inputSize  # feature dim
        self.queueSize = K  # memory size
        self.T = T
        self.index = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_class = num_class
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.w_pos = w_pos
        self.hard_iter = hard_iter
        self.num_hard = num_hard
        self.hard_negative_start = hard_negative_start

        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

        self.criterion = nn.CrossEntropyLoss()
        # number of positive
        if knn == -1:
            # default set
            self.knn = int(self.queueSize / num_class / 2)
        else:
            self.knn = knn

        # label for the labeled data
        self.label = nn.Parameter(torch.zeros(self.queueSize) - 1)
        self.label.requires_grad = False

    def forward(self, q, k, labels=None, epoch=0, labeled=False, la_memory=None):
        batchSize = q.shape[0]
        self.k_no_detach = k
        k = k.detach()
        self.epoch = epoch
        self.feat = q
        self.this_labels = labels
        self.k = k.detach()
        self.la_memory = la_memory

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        x = out
        x = x.squeeze()
        if labeled:
            loss = self.supervised_loss(x, self.label, labels)
        else:
            loss = self.ncl_loss(x)

        # update memory
        self.update_memory(batchSize, q, labels)

        return loss

    def supervised_loss(self, inputs, all_labels, la_labels):
        targets_onehot = torch.zeros(inputs.size()).to(self.device)
        for i in range(inputs.size(0)):
            this_idx = all_labels == la_labels[i].float()
            one_tensor = torch.ones(1).to(self.device)
            this_idx = torch.cat((one_tensor == 1, this_idx))
            ones_mat = torch.ones(torch.nonzero(this_idx).size(0)).to(self.device)
            weights = F.softmax(ones_mat, dim=0)
            targets_onehot[i, this_idx] = weights
        # targets_onehot[:, 0] = 0.2
        targets = targets_onehot.detach()
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def ncl_loss(self, inputs):

        targets = self.smooth_hot(inputs.detach().clone())

        if self.epoch < self.hard_negative_start:

            outputs = F.log_softmax(inputs, dim=1)
            loss = - (targets * outputs)
            loss = loss.sum(dim=1)

            loss = loss.mean(dim=0)

            return loss
        else:
            loss = self.ncl_hng_loss(self.feat, inputs, targets, self.memory.clone())
            return loss

    def smooth_hot(self, inputs):
        # Sort
        value_sorted, index_sorted = torch.sort(inputs[:, :], dim=1, descending=True)
        ldb = self.w_pos
        # ones_mat = torch.ones(inputs.size(0), self.knn).to(self.device)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)

        # weights = F.softmax(ones_mat, dim=1) * (1 - ldb)
        # targets_onehot.scatter_(1, index_sorted[:, 0:self.knn], weights)
        targets_onehot[:, 0] = float(ldb)

        return targets_onehot

    def ncl_hng_loss(self, feat, inputs, targets, memory):
        new_simi = []
        new_targets = []

        _, index_sorted_all = torch.sort(inputs[:, 1:], dim=1, descending=True)  # ignore first self-similarity
        _, index_sorted_all_all = torch.sort(inputs, dim=1, descending=True)  # consider all similarities

        if self.num_class == 5:
            num_neg = 50
        else:
            num_neg = 400

        for i in range(feat.size(0)):
            neg_idx = index_sorted_all[i, -num_neg:]
            la_memory = self.la_memory.detach().clone()
            neg_memory = memory[neg_idx].detach().clone()

            # randomly generate negative features
            new_neg_memory = []
            for j in range(self.hard_iter):
                rand_idx = torch.randperm(la_memory.size(0))
                this_new_neg_memory = (neg_memory * 1 + la_memory[rand_idx][:num_neg] * 2) / 3
                new_neg_memory.append(this_new_neg_memory)
                this_new_neg_memory = (neg_memory * 2 + la_memory[rand_idx][:num_neg] * 1) / 3
                new_neg_memory.append(this_new_neg_memory)
            new_neg_memory = torch.cat(new_neg_memory, dim=0)
            new_neg_memory = F.normalize(new_neg_memory)

            # select hard negative samples
            this_neg_simi = feat[i].view(1, -1).mm(new_neg_memory.t())
            value_sorted, index_sorted = torch.sort(this_neg_simi.view(-1), dim=-1, descending=True)
            this_neg_simi = this_neg_simi[0, index_sorted[:self.num_hard]]
            this_neg_simi = this_neg_simi / self.T

            targets_onehot = torch.zeros(this_neg_simi.size()).to(self.device)
            this_simi = torch.cat((inputs[i, index_sorted_all_all[i, :]].view(1, -1),
                                   this_neg_simi.view(1, -1)), dim=1)
            this_targets = torch.cat((targets[i, index_sorted_all_all[i, :]].view(1, -1),
                                      targets_onehot.view(1, -1)), dim=1)

            new_simi.append(this_simi)
            new_targets.append(this_targets)

        new_simi = torch.cat(new_simi, dim=0)
        new_targets = torch.cat(new_targets, dim=0)

        outputs = F.log_softmax(new_simi, dim=1)
        loss = - (new_targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)

        return loss

    def update_memory(self, batchSize, k, labels):
        # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)

            if labels is not None:
                self.label.index_copy_(0, out_ids, labels.float().detach().clone())

            self.index = (self.index + batchSize) % self.queueSize

def NCL_train(data, index, config, num_old, num_new, device):
    model, old_model = get_new_model(data, config, num_old, num_new, device)

    print ('Start Neighborhood Contrastive Learning:')
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=170, gamma=0.1)
    w_ncl_la, w_ncl_ulb = 0.1, 1.0
    ncl_ulb = NCLMemory(inputSize=config['hidden_size'], num_class=num_new).to(device)
    ncl_la = NCLMemory(inputSize=config['hidden_size'], num_class=num_old).to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    for epoch in range(200):
        model.train()
        exp_lr_scheduler.step()
        w = 5.0 * sigmoid_rampup(epoch, 50)

        out1, out2, feat = model(data.x, data.edge_index)
        model.l2norm = Normalize(2)
        tmp_feat = feat.view(feat.size(0), -1)
        feat_q = model.l2norm(tmp_feat)
        # feat_q = model.l2norm(feat)
        feat_k = feat_q + 0.2*torch.normal(0,torch.ones_like(feat_q)*feat_q.std()).to(device)

        # feat_bar, feat_k, output1_bar, output2_bar = model(x_bar, 'feat_logit')
        # prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)
        vice_out1 = out1 + 0.2*torch.normal(0,torch.ones_like(out1)*out1.std()).to(device)
        prob1, prob1_bar = F.softmax(out1[index.new_train_mask], dim=1), F.softmax(vice_out1[index.new_train_mask], dim=1)

        vice_out2 = out2 + 0.2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)
        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)

        rank_feat = (feat[index.new_train_mask]).detach()

        # cosine similarity with threshold
        feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
        tmp_distance_ori = torch.bmm(feat_row.view(feat_row.size(0), 1, -1), feat_col.view(feat_row.size(0), -1, 1))
        tmp_distance_ori = tmp_distance_ori.squeeze()
        target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
        target_ulb[tmp_distance_ori > 0.95] = 1

        prob1_ulb, _ = PairEnum(prob2)
        _, prob2_ulb = PairEnum(prob2_bar)

        # basic loss
        # loss_ce = criterion1(out1[index.old_train_mask], data.y[index.old_train_mask])
        loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
        consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
        # loss = loss_ce + loss_bce + w * consistency_loss
        loss = loss_bce + w * consistency_loss      # without label

        # NCL loss for unlabeled data
        # loss_ncl_ulb = ncl_ulb(feat_q[index.new_train_mask], feat_k[index.new_train_mask], data.y[index.new_train_mask], epoch, False, ncl_la.memory.clone().detach())
        loss_ncl_ulb = ncl_ulb(feat_q[index.new_train_mask], feat_k[index.new_train_mask], None, epoch, False)

        # NCL loss for labeled data
        # loss_ncl_la = ncl_la(feat_q[index.old_train_mask], feat_k[index.old_train_mask], data.y[index.old_train_mask], epoch, True)
        loss_ncl_la = ncl_la(feat_q[index.old_train_mask], feat_k[index.old_train_mask], None, epoch, False)
        if epoch > 0:
            loss += loss_ncl_ulb * w_ncl_ulb + loss_ncl_la * w_ncl_la
        else:
            loss += loss_ncl_la * w_ncl_la

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n=============== NCL for Dataset {} Final Test Accuracy ===============".format(config['dataset_name']))
    test_new_acc = incd_test(model, data, split="head1", sub_mask=index.new_test_mask)
    test_new_acc_2 = incd_test(model, data, split="head2", sub_mask=index.new_test_mask, num_old=config['num_old'])
    test_old_acc = incd_test(model, data, split="head1", sub_mask=index.old_test_mask)
    test_all_acc = incd_test(model, data, split="head1", sub_mask=index.all_test_mask)
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')
    print(f'head2: new = {test_new_acc_2:.4f}')

    model_dir = config['model_dir'] + "NCL_{}_hidden{}_{}_epoch{}.pth".format(config['model_name'], config['hidden_size'],config['dataset_name'], config['ncd_train_epochs'])
    # torch.save(model.state_dict(), model_dir)


# DTC-2019-ICCV baseline on GCN
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def feat2prob(feat, center, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(feat.unsqueeze(1) - center, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def init_prob_kmeans(old_model, data, sub_mask, config, device):
    torch.manual_seed(1)

    # cluster parameter initiate
    print ('Extract Labeled Feature')
    targets, feats = [], []
    old_model.eval()
    _, _, feat = old_model(data.x, data.edge_index)
    feats.append(feat[sub_mask].detach().clone().to(device))
    targets.append(data.y[sub_mask].detach().clone().to(device))
    feats = torch.cat(feats, dim=0).cpu().numpy()       # torch.Size([80, 16])
    targets = torch.cat(targets, dim=0).cpu().numpy()   # torch.Size([80, 16])

    # evaluate clustering performance
    from sklearn.decomposition import PCA
    n_clusters = config['hidden_size']
    pca = PCA(n_components=n_clusters)
    feats = pca.fit_transform(feats)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    # kmeans = KMeans(n_clusters=n_clusters, n_init=40)
    y_pred = kmeans.fit_predict(feats)
    probs = feat2prob(torch.from_numpy(feats), torch.from_numpy(kmeans.cluster_centers_))
    return kmeans.cluster_centers_, probs

def DTC_train(data, index, config, num_old, num_new, device):
    # init
    model, old_model = get_new_model(data, config, num_old, num_new, device)
    model.l2_classifier = False
    old_model.linear= Identity()
    # do not use old mask!!
    # init_centers, init_probs = init_prob_kmeans(old_model, data, index.new_train_mask, config, device)
    init_centers, init_probs = init_prob_kmeans(old_model, data, index.new_val_mask, config, device)
    # torch.Size([16, 16], torch.Size([80, 16])
    p_targets = target_distribution(init_probs) # torch.Size([80, 16]) or torch.Size([60, 16])

    from torch.nn.parameter import Parameter
    n_clusters = config['hidden_size']
    # n_clusters = config['num_old']
    model.center= Parameter(torch.Tensor(n_clusters, n_clusters))
    model.center.data = torch.tensor(init_centers).float().to(device)

    # warmup train
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    for epoch in range(10):
        model.train()
        _, _, feat = model(data.x, data.edge_index)
        prob = feat2prob(feat, model.center)    # prob: torch.Size([2708, 16])
        # loss = F.kl_div(prob[index.new_train_mask].log(), p_targets.float().to(device))
        loss = F.kl_div(prob[index.new_val_mask].log(), p_targets.float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # baseline train
    p_targets = target_distribution(init_probs)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    for epoch in range(100):
        model.train()
        _, _, feat = model(data.x, data.edge_index)
        prob = feat2prob(feat, model.center)    # torch.Size([2708, 16])
        # loss = F.kl_div(prob[index.new_train_mask].log(), p_targets.float().to(device))
        loss = F.kl_div(prob[index.new_val_mask].log(), p_targets.float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n=============== DTC for Dataset {} Final Test Accuracy ===============".format(config['dataset_name']))
    test_new_acc = incd_test(model, data, split="head1", sub_mask=index.new_test_mask)
    test_new_acc_2 = incd_test(model, data, split="head2", sub_mask=index.new_test_mask, num_old=config['num_old'])
    test_old_acc = incd_test(model, data, split="head1", sub_mask=index.old_test_mask)
    test_all_acc = incd_test(model, data, split="head1", sub_mask=index.all_test_mask)
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')
    print(f'head2: new = {test_new_acc_2:.4f}')

    model_dir = config['model_dir'] + "DTC_{}_hidden{}_{}_epoch{}.pth".format(config['model_name'], config['hidden_size'],config['dataset_name'], config['ncd_train_epochs'])
    # torch.save(model.state_dict(), model_dir)


# ResTune-2022-TNNLS baseline on GCN
class BCE_Res(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, P, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        #assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        #P = prob1.mul_(prob2)
        #P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_().mul_(torch.abs(simi))
        return neglogP.mean()

def ResTune_train(data, index, config, num_old, num_new, device):
    # init
    model = Res_GCN(data.num_features, hidden_channels=config['hidden_size'], model_name=config['model_name'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)

    # get model from warmup
    print("=============== Load warmup model...... ===============")
    warmup_model_dir = config['model_dir'] + \
                'warmup_{}_hidden{}_{}_epoch{}.pth'.format(config['model_name'], config['hidden_size'],
                                                    config['dataset_name'], config['pre_train_epochs'])
    # warmup_model_dir = config['model_dir'] + 'warmup_SAGE_hidden64_ogbn-arxiv_epoch200.pth'
    state_dict = torch.load(warmup_model_dir)
    model.load_state_dict(state_dict, strict=False)

    if config['model_name'] == 'GCN':
        old_model = GCN(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GraphSAGE':
        old_model = GraphSAGE(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GAT':
        old_model = GAT(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    elif config['model_name'] == 'GCNII':
        old_model = GCNII(data.num_features, hidden_channels=config['hidden_size'], num_labeled_classes=num_old, num_unlabeled_classes=num_new).to(device)
    old_model.load_state_dict(state_dict, strict=False)
    old_model.eval()

    save_weight = model.head1.weight.data.clone()   # save the weights of head-1
    save_bias = model.head1.bias.data.clone()       # save the bias of head-1
    model.head1 = nn.Linear(config['hidden_size'], num_old+num_new).to(device)       # replace the labeled-class only head-1 with the head-1-new include nodes for novel calsses

    model.head1.weight.data[:num_old] = save_weight       # put the old weights into the old part
    model.head1.bias.data[:] = torch.min(save_bias) - 1.    # put the bias
    model.head1.bias.data[:num_old] = save_bias

    if config['model_name'] == 'GCN':
        model.conv1.load_state_dict(old_model.conv1.state_dict())
        model.conv2.load_state_dict(old_model.conv2.state_dict())
        model.conv2_unlabel.load_state_dict(old_model.conv2.state_dict())
    elif config['model_name'] == 'GraphSAGE':
        model.conv1.load_state_dict(old_model.sage1.state_dict())
        model.conv2.load_state_dict(old_model.sage2.state_dict())
        model.conv2_unlabel.load_state_dict(old_model.sage2.state_dict())
    elif config['model_name'] == 'GAT':
        model.conv1.load_state_dict(old_model.gat1.state_dict())
        model.conv2.load_state_dict(old_model.gat2.state_dict())
        model.conv2_unlabel.load_state_dict(old_model.gat2.state_dict())

    # model.l2_classifier = False
    old_model.linear= Identity()
    init_centers, init_probs = init_prob_kmeans(old_model, data, index.new_train_mask, config, device)
    # init_centers, init_probs = init_prob_kmeans(old_model, data, index.new_val_mask, config, device)
    # torch.Size([16, 16], torch.Size([80, 16])
    p_targets = target_distribution(init_probs) # torch.Size([80, 16])

    from torch.nn.parameter import Parameter
    n_clusters = config['hidden_size']
    model.center= Parameter(torch.Tensor(n_clusters, n_clusters))
    model.center.data = torch.tensor(init_centers).float().to(device)

    bce_criterion = BCE_Res().to(device)
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    for epoch in range(100):
    # for epoch in range(600):
        # print(f"in epoch {epoch}")
        model.train()
        old_model.eval()

        # original codes: feat_label is feat, out_label is out1, out_pca out_bce is special out2
        out_label, out_pca, out_bce, feat_label = model(data.x, data.edge_index)
        # feat_unlabel = feat_unlabel.view(feat_unlabel.size(0), -1)
        out_pca_bar = out_pca + 0.2 * torch.normal(0, torch.ones_like(out_pca) * out_pca.std()).to(device)
        # prob_pca = feat2prob(feat, model.center)    # torch.Size([2708, 16])
        prob_pca = feat2prob(out_pca, model.center)
        prob_pca_bar = feat2prob(out_pca_bar, model.center)
        prob = F.softmax(out_bce, dim=1)

        sharp_loss = F.kl_div(prob[index.new_train_mask].log(), p_targets.float().to(device))
        # sharp_loss = F.kl_div(prob[index.new_val_mask].log(), p_targets.float().to(device))
        consistency_loss = F.mse_loss(prob_pca, prob_pca_bar)
        gc.collect()
        torch.cuda.empty_cache()

        # # rank loss
        # out_label = F.softmax(out_label, dim=1)
        # feat_copy = out_label.detach()
        # sim_mat = feat_copy.mm(feat_copy.t())

        # topk = 10
        # # topk=3
        # rank_idx_positive = torch.argsort(sim_mat, dim=1, descending=True)
        # rank_idx_positive = rank_idx_positive[:, :topk]
        # rank_idx_negative = torch.argsort(sim_mat, dim=1, descending=False)
        # rank_idx_negative = rank_idx_negative[:, :topk]
        # target_ulb = torch.zeros_like(sim_mat).float().to(device)
        # for i in range(data.x.size(0)):
        #     target_ulb[i, rank_idx_positive[i,:]] = 1
        #     target_ulb[i, rank_idx_negative[i, :]] = -1
        #     target_ulb[i, i] = 1

        # target_ulb = target_ulb.view(-1)
        # prob_mat = prob.mm(prob.t())
        # prob_mat = prob_mat.view(-1)

        # rank_loss = bce_criterion(prob_mat, target_ulb)
        gc.collect()
        torch.cuda.empty_cache()

        # LwF loss
        _, _, ref_output = old_model(data.x, data.edge_index)
        # soft_target = F.softmax(ref_output[index.new_val_mask] / 2, dim=1)
        soft_target = F.softmax(ref_output[index.new_train_mask] / 2, dim=1)
        new_output = old_model.linear(feat_label)
        # logp = F.log_softmax(new_output[index.new_val_mask] / 2, dim=1)
        logp = F.log_softmax(new_output[index.new_train_mask] / 2, dim=1)
        reg_loss = -torch.mean(torch.sum(soft_target * logp, dim=1))
        gc.collect()
        torch.cuda.empty_cache()

        # total loss
        # loss = sharp_loss + consistency_loss + rank_loss + 0.1 * reg_loss
        # loss = sharp_loss + consistency_loss + 0.1 * reg_loss
        loss = 0.1*sharp_loss + 0.1*consistency_loss + 0.01 * reg_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()
        val_acc, test_acc = test_while_training(model, data, index.new_val_mask, index.new_test_mask)
        old_val_acc, old_test_acc = test_while_training(model, data, index.old_val_mask, index.old_test_mask)
        all_val_acc, all_test_acc = test_while_training(model, data, data.val_mask, data.test_mask)

        # Start to log and print
        if epoch % config['print_every_epochs'] == 0:
            format_str = 'Epoch {:03d}, train_loss: {:.4f} | new_val_acc: {:.4f}, new_test_acc: {:.4f}, old_val_acc: {:.4f}, old_test_acc: {:.4f}, all_val_acc: {:.4f}, all_test_acc: {:.4f}'.format(epoch, loss, val_acc, test_acc, old_val_acc, old_test_acc, all_val_acc, all_test_acc)
            print(format_str)

    print("\n=============== ResTune for Dataset {} Final Test Accuracy ===============".format(config['dataset_name']))
    model.eval()
    out1, _, _, _ = model(data.x, data.edge_index)
    pred = out1.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[index.new_test_mask] == (data.y[index.new_test_mask])
    test_new_acc = int(test_correct.sum()) / int(index.new_test_mask.sum())

    test_correct = pred[index.old_test_mask] == (data.y[index.old_test_mask])
    test_old_acc = int(test_correct.sum()) / int(index.old_test_mask.sum())

    test_correct = pred[index.all_test_mask] == (data.y[index.all_test_mask])
    test_all_acc = int(test_correct.sum()) / int(index.all_test_mask.sum())
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')

    model_dir = config['model_dir'] + "ResTune_{}_hidden{}_{}_epoch{}.pth".format(config['model_name'], config['hidden_size'],config['dataset_name'], config['ncd_train_epochs'])
    # torch.save(model.state_dict(), model_dir)

## ResTune test while training
def test_while_training(model, data, val_mask, test_mask):
    model.eval()
    out, _, _, _ = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    val_correct = pred[val_mask] == data.y[val_mask]
    val_acc = int(val_correct.sum()) / int(val_mask.sum())
    test_correct = pred[test_mask] == data.y[test_mask]
    test_acc = int(test_correct.sum()) / int(test_mask.sum())

    return val_acc, test_acc

# TODO
# other abla
def LwF_woProto_woKD_train(model, old_model, data, index, config, device, old_mean, old_sig):
    from torch.optim import lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(1, config['ncd_train_epochs'] + 1):
        model.train()
        optimizer.zero_grad()  # Clear gradients.

        out1, out2, feat = model(data.x, data.edge_index)  # Perform a single forward pass.
        vice_out2 = out2 + 0.2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)
        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)

        loss_novel  = get_loss_bce(prob2, prob2_bar, feat[index.new_train_mask], device) + \
                      get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],
                                  num_old=config['num_old'], rampup_length=config['rampup_length'],
                                  increment_coefficient=config['increment_coefficient']) + \
                      get_loss_mse(prob2, prob2_bar, config, epoch)
        loss_past = get_loss_replay(model, old_mean, old_sig,
                                    num_old=config['num_old'], lambda_proto=config['lambda_proto'], device=device) + \
                    get_loss_kd(old_model, feat, data, w_kd=config['w_kd'])
        # loss = loss_novel + loss_past
        loss = loss_novel * 1.2 + loss_past*0.5

        loss.backward()   # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    print("\n=============== NCL for Dataset {} Final Test Accuracy ===============".format(config['dataset_name']))
    test_new_acc = incd_test(model, data, split="head1", sub_mask=index.new_test_mask)
    test_new_acc_2 = incd_test(model, data, split="head2", sub_mask=index.new_test_mask, num_old=config['num_old'])
    test_old_acc = incd_test(model, data, split="head1", sub_mask=index.old_test_mask)
    test_all_acc = incd_test(model, data, split="head1", sub_mask=index.all_test_mask)
    print(f'head1: old = {test_old_acc:.4f}, new = {test_new_acc:.4f}, all = {test_all_acc:.4f}')
    print(f'head2: new = {test_new_acc_2:.4f}')

# TODO
def LwF_woKD_train(model, old_model, data, index, config, device, old_mean, old_sig):
    # without KD
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(1, config['ncd_train_epochs'] + 1):
        model.train()
        optimizer.zero_grad()  # Clear gradients.

        out1, out2, feat = model(data.x, data.edge_index)  # Perform a single forward pass.
        vice_out2 = out2 + 0.2*torch.normal(0,torch.ones_like(out2)*out2.std()).to(device)
        prob2, prob2_bar = F.softmax(out2[index.new_train_mask], dim=1), F.softmax(vice_out2[index.new_train_mask], dim=1)

        loss_novel  = get_loss_bce(prob2, prob2_bar, feat[index.new_train_mask], device) + \
                      get_loss_ce(epoch, out1[index.new_train_mask], out2[index.new_train_mask],
                                  num_old=config['num_old'], rampup_length=config['rampup_length'],
                                  increment_coefficient=config['increment_coefficient']) + \
                      get_loss_mse(prob2, prob2_bar, config, epoch)
        loss_past = get_loss_replay(model, old_mean, old_sig,
                                    num_old=config['num_old'], lambda_proto=config['lambda_proto'], device=device) + \
                    get_loss_kd(old_model, feat, data, w_kd=config['w_kd'])
        # loss = loss_novel + loss_past
        loss = loss_novel * 1.2 + loss_past*0.5

        loss.backward()   # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    print("\n=============== NCL for Dataset {} Final Test Accuracy ===============".format(config['dataset_name']))
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



def freeze_layers(model, layer_names, freeze=True):
    from collections.abc import Iterable
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def unfreeze_layers(model, layer_names):
    freeze_layers(model, layer_names, False)

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

    # Dataloader creation
    data = dataset[0]

    print("================== Getting subdata split idx...... ==================")
    index = type('IndexObject', (object,), {})()
    # index.old_train_mask, index.old_test_mask, index.old_val_mask = load_subdata(data, num_old=num_old, split="old")
    # index.new_train_mask, index.new_test_mask, index.new_val_mask = load_subdata(data, num_old=num_old, split="new")
    # index.all_train_mask, index.all_test_mask, index.all_val_mask = load_subdata(data, num_old=num_old, split="old+new")
    if config['dataset_name'] == 'ogbn-arxiv':
        data.y = data.y.T[0]
        index.old_train_mask, index.old_test_mask, index.old_val_mask = load_ogbdata(data, num_old=num_old, split="old", split_idx=dataset.get_idx_split())
        index.new_train_mask, index.new_test_mask, index.new_val_mask = load_ogbdata(data, num_old=num_old, split="new", split_idx=dataset.get_idx_split())
        index.all_train_mask, index.all_test_mask, index.all_val_mask = load_ogbdata(data, num_old=num_old, split="old+new", split_idx=dataset.get_idx_split())
    else:
        index.old_train_mask, index.old_test_mask, index.old_val_mask = load_subdata(data, num_old=num_old, split="old")
        index.new_train_mask, index.new_test_mask, index.new_val_mask = load_subdata(data, num_old=num_old, split="new")
        index.all_train_mask, index.all_test_mask, index.all_val_mask = load_subdata(data, num_old=num_old, split="old+new")


    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # device = "cpu"
    data = data.to(device)


    print("=============== Start incd-training in comparison for Dataset {} ...... ===============".format(config['dataset_name']))

    # model, old_model = get_new_model(data, config, num_old, num_new, device)
    # old_mean, old_sig = cal_mean_sig(model, data, index.old_train_mask, config['hidden_size'], num_old, device)
    # incd_train_SGD(model, old_model, data, index, config, device, old_mean, old_sig)

    # DTC_train(data, index, config, num_old, num_new, device)
    # AutoNovel_train(data, index, config, num_old, num_new, device)
    NCL_train(data, index, config, num_old, num_new, device)
    ResTune_train(data, index, config, num_old, num_new, device)

    print('\nDone!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='path to the config file')
    args = parser.parse_args()

    main(args)