import argparse
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

LR = 2e-5
SEED = 17
SEEDLIST = [17]

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='train path')
    parser.add_argument('--dev_path', type=str, required=True, help='dev path')
    parser.add_argument('--test_path', type=str, required=True, help='test path')
    parser.add_argument('--label_map_path', type=str, required=True, help='label_map path')
    parser.add_argument('--best_model_path', type=str, help='best model path')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--max_seq_len', type=int, default=512, help='max sequence length')
    parser.add_argument('--resp_source_epoch', type=int, default=10, help='resp source epoch')
    parser.add_argument('--resp_source_batch_size', type=int, default=16, help='resp source batch size')
    parser.add_argument('--resp_epoch', type=int, default=10, help='resp epoch')
    parser.add_argument('--resp_batch_size', type=int, default=5, help='resp batch size')
    parser.add_argument('--cls_epoch', type=int, default=15, help='cls epoch')
    parser.add_argument('--cls_batch_size', type=int, default=10, help='cls batch size')
    parser.add_argument('--accum_iter', type=int, default=4, help='accum iter')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda')
    parser.add_argument('--temp', type=float, default=0.8, help='temp')
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    parser.add_argument('--task', type=str, required=True, help='task')
    return parser.parse_args()

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def write_log(filename, log):
    with open(filename, 'a+') as f:
        print(log, file=f)

def colorprint(text, color):
    color_dict = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'grey': '\033[98m',
        'black': '\033[99m',
        'reset': '\033[0m',
    }
    print(color_dict[color] + text + color_dict['reset'])

def compute_topk_acc(label, pred):
    cnt = 0
    for l, p in zip(label, pred):
        if l in p:
            cnt += 1
    return round(cnt / len(label), 4)

def mean_reciprocal_rank(labels, predictions):
    references = []
    for label, pred in zip(labels, predictions):
        references.append([0 if p != label else 1 for p in pred])
    ranks = []
    for rf in references:
        rank = 0
        for i, r in enumerate(rf):
            if r == 1:
                rank = i + 1
                break
        ranks.append(1.0 / rank if rank != 0 else 0)
    return round(sum(ranks) * 1.0 / len(ranks), 4)

def compute_metric(label, pred, average='macro', f1_average=None):
    f1_average = average if f1_average is None else f1_average
    precision = round(precision_score(label, pred, average=average, zero_division=1), 4)
    recall = round(recall_score(label, pred, average=average, zero_division=1), 4)
    f1 = round(f1_score(label, pred, average=f1_average, zero_division=1), 4)
    acc = round(accuracy_score(label, pred), 4)
    return precision, recall, f1, acc

def plot_confusion_matrix(label, pred):
    _, ax = plt.subplots(figsize=(25, 25))
    disp = ConfusionMatrixDisplay.from_predictions(label, pred, cmap=plt.cm.Blues, normalize='true', ax=ax)
    plt.savefig('confusion_matrix.png')
