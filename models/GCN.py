import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GCN2Conv, JumpingKnowledge

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_labeled_classes=4, num_unlabeled_classes=3):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.head1 = Linear(hidden_channels, num_labeled_classes)
        self.head2 = Linear(hidden_channels, num_unlabeled_classes)

        self.l2_classifier = False

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        if self.l2_classifier:
            out1 = self.head1(F.normalize(x, dim=-1)) / 0.1
        else:
            out1 = self.head1(x)
        out2 = self.head2(x)

        return out1, out2, x


class SAGENet(torch.nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=3, num_old=4, num_new=3):
        super(SAGENet, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(SAGEConv(dataset.num_node_features, hidden))
        self.bns.append(torch.nn.BatchNorm1d(hidden))

        for i in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(torch.nn.BatchNorm1d(hidden))

        # self.convs.append(SAGEConv(hidden, dataset.num_classes))
        self.convs.append(SAGEConv(hidden, hidden))

        # self.head1 = SAGEConv(hidden, num_old)
        # self.head2 = SAGEConv(hidden, num_new)
        self.head1 = Linear(hidden, num_old)
        self.head2 = Linear(hidden, num_new)
        self.l2_classifier = False

    def forward(self, x, adjs):
        for i, (adj_t, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), adj_t)
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        out1 = F.log_softmax(self.head1(x), dim=1)
        out2 = F.log_softmax(self.head2(x), dim=1)

        x = F.log_softmax(x, dim=1)

        return out1, out2, x

    def inference(self, x_all, subgraph_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                adj_t, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), adj_t)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = F.relu(x)
                # xs.append(x.cpu())
                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        x_all = self.head1(x_all).to('cpu')

        return x_all


# ResTune-2022-TNNLS baseline on GCN
class Res_GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, model_name, num_labeled_classes=4, num_unlabeled_classes=3):
        super().__init__()
        torch.manual_seed(1234567)

        if model_name == 'GCN':
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv2_unlabel = GCNConv(hidden_channels, hidden_channels)
        elif model_name == 'GraphSAGE':
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv2_unlabel = SAGEConv(hidden_channels, hidden_channels)
        elif model_name == 'GAT':
            self.conv1 = GATv2Conv(num_features, hidden_channels, heads=8)
            self.conv2 = GATv2Conv(hidden_channels*8, hidden_channels, heads=1)
            self.conv2_unlabel = GATv2Conv(hidden_channels*8, hidden_channels, heads=1)

        self.head1 = Linear(hidden_channels, num_labeled_classes)
        # self.head2_pca = Linear(hidden_channels, num_unlabeled_classes)
        # self.head2_bce = Linear(hidden_channels, num_unlabeled_classes)
        self.head2_pca = Linear(hidden_channels, hidden_channels)
        self.head2_bce = Linear(hidden_channels, hidden_channels)

        # self.l2_classifier = False

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        feat_label = self.conv2(x, edge_index)
        # feat_label = feat_label.view(feat_label.size(0), -1)

        feat_unlabel = self.conv2_unlabel(x, edge_index)
        # feat_unlabel = feat_unlabel.view(feat_unlabel.size(0), -1)

        feat = feat_label + feat_unlabel

        out1 = self.head1(F.normalize(feat_label, dim=-1)) / 0.1
        # out2 = self.head2(x)
        out_pca = self.head2_pca(feat)
        out_bce = self.head2_bce(feat_unlabel)

        return out1, out_pca, out_bce, feat_label


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, num_features, hidden_channels, num_labeled_classes=4, num_unlabeled_classes=3, heads=8):
        super().__init__()
        # self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        # self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.gat1 = GATv2Conv(num_features, hidden_channels, heads=heads)
        self.gat2 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=1)
        self.head1 = Linear(hidden_channels, num_labeled_classes)
        self.head2 = Linear(hidden_channels, num_unlabeled_classes)
        self.l2_classifier = False

    # def forward(self, x, edge_index):
    #     h = F.dropout(x, p=0.6, training=self.training)
    #     h = self.gat1(h, edge_index)
    #     h = F.elu(h)
    #     h = F.dropout(h, p=0.6, training=self.training)
    #     h = self.gat2(h, edge_index)
    #     return h, F.log_softmax(h, dim=1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)

        if self.l2_classifier:
            out1 = self.head1(F.normalize(x, dim=-1)) / 0.1
        else:
            out1 = self.head1(x)
        out2 = self.head2(x)

        return out1, out2, x


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""
    def __init__(self, num_features, hidden_channels, num_labeled_classes=4, num_unlabeled_classes=3):
        super().__init__()
        self.sage1 = SAGEConv(num_features, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.head1 = Linear(hidden_channels, num_labeled_classes)
        self.head2 = Linear(hidden_channels, num_unlabeled_classes)
        self.l2_classifier = False

    # def forward(self, x, edge_index):
    #     h = self.sage1(x, edge_index).relu()
    #     h = F.dropout(h, p=0.5, training=self.training)
    #     h = self.sage2(h, edge_index)
    #     return F.log_softmax(h, dim=1)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage2(x, edge_index)

        if self.l2_classifier:
            out1 = self.head1(F.normalize(x, dim=-1)) / 0.1
        else:
            out1 = self.head1(x)
        out2 = self.head2(x)

        return out1, out2, x


class GCNII(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_labeled_classes=4, num_unlabeled_classes=3, num_layers=4):
        super(GCNII, self).__init__()

        self.num_layers = num_layers  # 层数
        self.identity = torch.nn.Identity()

        # self.conv1 = GCNConv(num_features, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCN2Conv(num_features, alpha=0.1))
        self.conv2 = GCNConv(num_features, hidden_channels)

        self.head1 = Linear(hidden_channels, num_labeled_classes)
        self.head2 = Linear(hidden_channels, num_unlabeled_classes)
        self.l2_classifier = False

    def forward(self, x, edge_index):
        # x = torch.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x_0 = x
        for i in range(self.num_layers - 1):
            # print("gcnii layer:", i)
            x = self.convs[i](x, x_0, edge_index)
            x = torch.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x_0 = x

        x = self.conv2(x, edge_index)

        if self.l2_classifier:
            out1 = self.head1(F.normalize(x, dim=-1)) / 0.1
        else:
            out1 = self.head1(x)
        out2 = self.head2(x)
        return out1, out2, x


# GAT
class GCN_list(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_labeled_classes=4, num_unlabeled_classes=3):
        super().__init__()
        torch.manual_seed(1234567)
        self.gat_layers = torch.nn.ModuleList()

        # self.gat_layers.append(GCNConv(num_features, hidden_channels))
        # self.gat_layers.append(GCNConv(hidden_channels, hidden_channels))
        # self.gat_layers.append(GCNConv(hidden_channels, num_labeled_classes))
        # GAT
        # self.gat_layers.append(GATv2Conv(num_features, hidden_channels, heads=8))
        # self.gat_layers.append(GATv2Conv(hidden_channels*8, hidden_channels, heads=1))
        # self.gat_layers.append(GATv2Conv(hidden_channels, num_labeled_classes, heads=1))

        self.gat_layers.append(SAGEConv(num_features, hidden_channels))
        self.gat_layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.gat_layers.append(SAGEConv(hidden_channels, num_labeled_classes))

    def forward(self, x, edge_index):
        x = self.gat_layers[0](x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat_layers[1](x, edge_index)
        out = self.gat_layers[2](x, edge_index)
        return out, out, x


if __name__ == '__main__':
    model = GCN(1433, 16)
    print(model)
