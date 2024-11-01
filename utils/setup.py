import argparse
import yaml
import torch
# import gc
# from ogb.nodeproppred import Evaluator


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='whether open multiple run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("******************** MODEL CONFIGURATION ********************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} ==> {}".format(keystr, val))
    print("********************* END CONFIGURATION *********************")

# import sys
# class Logger(object):
#     def __init__(self, log_file):
#         self.terminal = sys.stdout
#         self.log = open(log_file, "a")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()

#     def flush(self):
#         pass

def write_to_file(format_str, log_file_name):
    with open(log_file_name, 'a') as f:
        f.write(format_str)
    f.close()



def test_while_training(model, data, val_mask, test_mask):
    model.eval()
    out, _, _ = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    val_correct = pred[val_mask] == data.y[val_mask]
    val_acc = int(val_correct.sum()) / int(val_mask.sum())
    test_correct = pred[test_mask] == data.y[test_mask]
    test_acc = int(test_correct.sum()) / int(test_mask.sum())

    return val_acc, test_acc


@torch.no_grad()
def test_while_training_in_batch(model, index, x, y, subgraph_loader, split="old"):
    model.eval()

    out = model.inference(x, subgraph_loader)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []

    if split == "new":
        for mask in [index.new_train_mask, index.new_val_mask, index.new_test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
    elif split == "old":
        for mask in [index.old_train_mask, index.old_val_mask, index.old_test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


@torch.no_grad()
def test_while_training_in_batch_idx(model, x, y, subgraph_loader, evaluator, index, split="old"):
    model.eval()

    out = model.inference(x, subgraph_loader)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if split == "new":
        train_acc = evaluator.eval({
            'y_true': y_true[index.new_train_idx],
            'y_pred': y_pred[index.new_train_idx],
        })['acc']
        val_acc = evaluator.eval({
            'y_true': y_true[index.new_val_idx],
            'y_pred': y_pred[index.new_val_idx],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[index.new_test_idx],
            'y_pred': y_pred[index.new_test_idx],
        })['acc']
    elif split == "old":
        train_acc = evaluator.eval({
            'y_true': y_true[index.old_train_idx],
            'y_pred': y_pred[index.old_train_idx],
        })['acc']
        val_acc = evaluator.eval({
            'y_true': y_true[index.old_val_idx],
            'y_pred': y_pred[index.old_val_idx],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[index.old_test_idx],
            'y_pred': y_pred[index.old_test_idx],
        })['acc']

    return train_acc, val_acc, test_acc
