import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ramps import sigmoid_rampup


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 计算旧类的均值和方差
def cal_mean_sig(model, data, sub_mask, hidden, num_old=4, device='cuda'):
    all_feat = []
    all_labels = []

    # class_sig = torch.zeros(args.num_labeled_classes, 512).to(device)

    class_mean = torch.zeros(num_old, hidden).to(device)
    class_sig = torch.zeros(num_old, hidden).to(device)

    print ('Extract Labeled Feature')
    model.eval()
    _, _, feat = model(data.x, data.edge_index)
    # _, _, feat = model(data.x, data.x, data.edge_index)       ## GCNII

    all_feat.append(feat[sub_mask].detach().clone().to(device))
    all_labels.append(data.y[sub_mask].detach().clone().to(device))

    all_feat = torch.cat(all_feat, dim=0).to(device)        # torch.Size([80, 16])
    all_labels = torch.cat(all_labels, dim=0).to(device)    # torch.Size([80])

    print ('Calculate Labeled Mean-Var')
    for i in range(num_old):       # 分别计算supervised training阶段的均值和标准差
        this_feat = all_feat[all_labels==i]
        this_mean = this_feat.mean(dim=0)
        this_var = this_feat.var(dim=0)
        class_mean[i, :] = this_mean
        class_sig[i, :] = (this_var + 1e-5).sqrt()
    print ('Finish')

    class_mean, class_sig = class_mean.to(device), class_sig.to(device)

    return class_mean, class_sig


def cal_mean_sig_in_batch(model, old_train_loader, data, sub_mask, config, device='cuda'):
    all_feat, all_labels = [], []
    num_old, hidden = config['num_old'], config['hidden_size']
    class_mean = torch.zeros(num_old, hidden).to(device)
    class_sig = torch.zeros(num_old, hidden).to(device)

    print ('Extract Labeled Feature')
    if config['dataset_name'] == 'ogbn-arxiv':
        x, y = data.x.to(device), data.y[sub_mask].T[0].to(device)
    else:
        x, y = data.x.to(device), data.y[sub_mask].to(device)
    # Cora: x=torch.Size([2708, 1433]), y=torch.Size([80])
    # arxiv: x=torch.Size([169343, 128]), y=torch.Size([50293])
    model.eval()
    for batch_size, n_id, adjs in old_train_loader:
        adjs = [adj.to(device) for adj in adjs]
        _, _, feat = model(x[n_id], adjs)

        all_feat.append(feat.detach().clone().to(device))
    all_labels.append(y.detach().clone().to(device))

    all_feat = torch.cat(all_feat, dim=0).to(device)        # torch.Size([80, 16])
    all_labels = torch.cat(all_labels, dim=0).to(device)    # torch.Size([80])

    print ('Calculate Labeled Mean-Var')
    for i in range(num_old):
        this_feat = all_feat[all_labels==i]
        this_mean = this_feat.mean(dim=0)
        this_var = this_feat.var(dim=0)
        class_mean[i, :] = this_mean
        class_sig[i, :] = (this_var + 1e-5).sqrt()
    print ('Finish')
    class_mean, class_sig = class_mean.to(device), class_sig.to(device)

    return class_mean, class_sig


# 工具代码 计算方差和均值
def Generate_Center(model, labeled_train_loader, args, device):
    all_feat = []
    all_labels = []

    class_mean = torch.zeros(args.num_labeled_classes, 512).cuda()
    class_sig = torch.zeros(args.num_labeled_classes, 512).cuda()

    print ('Extract Labeled Feature')
    for epoch in range(1):
        model.eval()
        for batch_idx, (x, label, idx) in enumerate(labeled_train_loader):
            x, label = x.to(device), label.to(device)

            output1, output2, feat = model(x)

            all_feat.append(feat.detach().clone().cuda())
            all_labels.append(label.detach().clone().cuda())

    all_feat = torch.cat(all_feat, dim=0).cuda()
    all_labels = torch.cat(all_labels, dim=0).cuda()

    print ('Calculate Labeled Mean-Var')
    for i in range(args.num_labeled_classes):       # 分别计算supervised training阶段的均值和标准差
        this_feat = all_feat[all_labels==i]
        this_mean = this_feat.mean(dim=0)
        this_var = this_feat.var(dim=0)
        class_mean[i, :] = this_mean
        class_sig[i, :] = (this_var + 1e-5).sqrt()
    print ('Finish')

    class_mean, class_sig, class_cov = class_mean.cuda(), class_sig.cuda(), 0

    return class_mean, class_sig, class_cov


def sample_features(class_mean, class_sig, num_old=4, device='cuda'):
    feats = []
    labels = []

    for i in range(num_old):
        dist = torch.distributions.Normal(class_mean[i], class_sig.mean(dim=0))     # 按均值mu和标准差sigma归一化
        this_feat = dist.sample((2,))
        this_label = torch.ones(this_feat.size(0)) * i
        # print(f'this_feat: {this_feat}, this_label: {this_label}')

        feats.append(this_feat.to(device))
        labels.append(this_label.to(device))

    feats = torch.cat(feats, dim=0)             # torch.Size([8, 16])
    labels = torch.cat(labels, dim=0).long()    # torch.Size([8])

    return feats, labels


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x  会用来计算成对相似性
    assert x.ndimension() == 2, 'Input dimension must be 2'

    # x.shape = torch.Size([40648, 16])
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        # dim 0: # sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


## old_version
# def rank_statistics(output2, feat, topk=3, device='cuda'):
#     # use softmax to get the probability distribution for each head
#     prob2 = F.softmax(output2, dim=1)
#     # print(f'prob2.shape = {prob2.shape}')   # torch.Size([60, 3])

#     # first cut the gradient propagation of the feat
#     rank_feat = (feat).detach()
#     # print(f'rank_feat: {rank_feat.shape}')  # torch.Size([60, 16])

#     rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
#     # print(f'rank_idx = {rank_idx.shape}') # arxiv: torch.Size([40648, 16])

#     rank_idx1, rank_idx2 = PairEnum(rank_idx)
#     rank_idx1, rank_idx2 = rank_idx1[:, :topk], rank_idx2[:, :topk]

#     rank_idx1, _ = torch.sort(rank_idx1, dim=1)
#     rank_idx2, _ = torch.sort(rank_idx2, dim=1)

#     rank_diff = rank_idx1 - rank_idx2
#     rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
#     target_ulb = torch.ones_like(rank_diff).float().to(device)
#     target_ulb[rank_diff > 0] = -1

#     # get the probability distribution of the prediction for head-2
#     prob1_ulb, prob2_ulb = PairEnum(prob2)

#     return prob1_ulb, prob2_ulb, target_ulb

# topk = 1,2,3,5,7,10,15,20,30,50
# def rank_statistics(prob2, prob2_bar, feat, topk=10, device='cuda'):
def rank_statistics(prob2, prob2_bar, feat, topk=3, device='cuda'):
    # first cut the gradient propagation of the feat
    rank_feat = (feat).detach()
    # print(f'rank_feat: {rank_feat.shape}')  # torch.Size([60, 16])

    rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
    # print(f'rank_idx = {rank_idx.shape}') # arxiv: torch.Size([40648, 16])

    rank_idx1, rank_idx2 = PairEnum(rank_idx)
    rank_idx1, rank_idx2 = rank_idx1[:, :topk], rank_idx2[:, :topk]

    rank_idx1, _ = torch.sort(rank_idx1, dim=1)
    rank_idx2, _ = torch.sort(rank_idx2, dim=1)

    rank_diff = rank_idx1 - rank_idx2
    rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
    target_ulb = torch.ones_like(rank_diff).float().to(device)
    target_ulb[rank_diff > 0] = -1

    # get the probability distribution of the prediction for head-2
    prob1_ulb, _ = PairEnum(prob2)
    _, prob2_ulb = PairEnum(prob2_bar)

    return prob1_ulb, prob2_ulb, target_ulb


class BCE(torch.nn.Module):
    eps = 1e-7      # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        # print(f'P = {P}')
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

## old_version
# def get_loss_bce(output2, feat, device="cuda"):
#     # loss_bce = torch.tensor(0.0)
#     criterion = BCE()
#     prob1_ulb, prob2_ulb, target_ulb = rank_statistics(output2, feat, device=device)
#     # print(f'p1:{prob1_ulb}, shape={prob1_ulb.shape}')         # torch.Size([3600, 3])
#     # print(f'p2:{prob2_ulb}, shape={prob2_ulb.shape}')         # torch.Size([3600, 3])
#     # print(f'y_ulb:{target_ulb}, shape={target_ulb.shape}')    # torch.Size([3600])

#     loss_bce = criterion(prob1_ulb, prob2_ulb, target_ulb)
#     return loss_bce

def get_loss_bce(prob2, prob2_bar, feat, device="cuda"):
    # loss_bce = torch.tensor(0.0)
    criterion = BCE()
    prob1_ulb, prob2_ulb, target_ulb = rank_statistics(prob2, prob2_bar, feat, device=device)
    # print(f'p1:{prob1_ulb}, shape={prob1_ulb.shape}')         # torch.Size([3600, 3])
    # print(f'p2:{prob2_ulb}, shape={prob2_ulb.shape}')         # torch.Size([3600, 3])
    # print(f'y_ulb:{target_ulb}, shape={target_ulb.shape}')    # torch.Size([3600])

    loss_bce = criterion(prob1_ulb, prob2_ulb, target_ulb)
    return loss_bce


def get_loss_ce(epoch, out1, out2, num_old=4, rampup_length=50, increment_coefficient=0.01):
    criterion = nn.CrossEntropyLoss()
    w = sigmoid_rampup(epoch, rampup_length) * increment_coefficient * 8
    # w = sigmoid_rampup(epoch, rampup_length) * increment_coefficient
    label = (out2).detach().max(1)[1] + num_old
    # print(f'out2_detach = {(out2[sub_mask]).detach().max(1)[1]}')
    # print(f'data.y = {data.y[sub_mask]-4}')
    # print(f'out: {out1[sub_mask].shape}')   # out: torch.Size([60, 7])
    # print(f'label: {label.shape}')  # label: torch.Size([60])

    loss_ce = w * criterion(out1, label)
    # print("beta1 = ", w)
    return loss_ce


def get_loss_mse(prob2, prob2_bar, config, epoch):
    # print("beta2=", config['rampup_coefficient'] * sigmoid_rampup(epoch, config['rampup_length']))
    return F.mse_loss(prob2, prob2_bar) * config['rampup_coefficient'] * sigmoid_rampup(epoch, config['rampup_length'])


def forward_feat(self, feat):
    out = feat
    # out = F.relu(out)  # add ReLU to benifit ranking
    if self.l2_classifier:
        out1 = self.head1(F.normalize(out, dim=-1))
    else:
        out1 = self.head1(out)
    # out2 = self.head2(out)
    return out1


# def get_loss_replay(model, data, sub_mask, hidden=16, num_old=4, lambda_proto=1.0, device="cuda"):
#     criterion = nn.CrossEntropyLoss()
#     old_mean, old_sig = cal_mean_sig(model, data, sub_mask, hidden, num_old, device)
#     # print(f'old_mean: {old_mean.shape}, old_sig: {old_sig.shape}')     # torch.Size([num_old, 16])
#     old_feats, old_labels = sample_features(old_mean, old_sig, num_old, device)

#     # labeled_output1 = model.forward_feat(labeled_feats)
#     old_out1 = model.head1(F.normalize(old_feats, dim=-1))
#     loss_replay = lambda_proto * criterion(old_out1, old_labels)

#     # loss_replay = torch.tensor(0.0)
#     return loss_replay
def get_loss_replay(model, old_mean, old_sig, num_old=4, lambda_proto=1.0, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    # print(f'old_mean: {old_mean.shape}, old_sig: {old_sig.shape}')     # torch.Size([num_old, 16])
    old_feats, old_labels = sample_features(old_mean, old_sig, num_old, device)

    # labeled_output1 = model.forward_feat(labeled_feats)
    if model.l2_classifier:
        old_out1 = model.head1(F.normalize(old_feats, dim=-1))
    else:
        old_out1 = model.head1(old_feats)
    loss_replay = lambda_proto * criterion(old_out1, old_labels)

    # loss_replay = torch.tensor(0.0)
    return loss_replay


def get_loss_kd(old_model, feat, data, w_kd=10):
    old_model.eval()
    _, _, old_feat = old_model(data.x, data.edge_index)
    size_1, size_2 = old_feat.size()
    loss_kd = torch.dist(F.normalize(old_feat.view(size_1 * size_2, 1), dim=0), F.normalize(feat.view(size_1 * size_2, 1), dim=0)) * w_kd

    # loss_kd = torch.tensor(0.0)
    return loss_kd

# def get_loss_kd(submask, old_model, feat, data, w_kd=10):
#     old_model.eval()
#     _, _, old_feat = old_model(data.x, data.edge_index)

#     old_feat, feat = old_feat[submask], feat[submask]
#     size_1, size_2 = old_feat.size()
#     loss_kd = torch.dist(F.normalize(old_feat.view(size_1 * size_2, 1), dim=0), F.normalize(feat.view(size_1 * size_2, 1), dim=0)) * w_kd

#     return loss_kd


def get_loss_kd_in_batch(old_model, feat, x, n_id, adjs, w_kd=10):
    old_model.eval()
    _, _, old_feat = old_model(x[n_id], adjs)
    size_1, size_2 = old_feat.size()
    loss_kd = torch.dist(F.normalize(old_feat.view(size_1 * size_2, 1), dim=0), F.normalize(feat.view(size_1 * size_2, 1), dim=0)) * w_kd
    # loss_kd = torch.tensor(0.0)
    return loss_kd
