from torch.utils.data import DataLoader
from dataHelper import collate, get_type_dataset
from progressbar import ProgressBar
import torch.optim as optim
import torch.nn as nn
import argparse
from itertools import count
from utils import cal_metrics
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_path')
args = parser.parse_args()

def inference():
    model_path = args.model_path
    model = torch.load(model_path)
    eps = 1e-15
    ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=True, ds='new_ds', half=False)
    train_loader = DataLoader(ds.train_ds, batch_size=1, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)
    user_id = 0
    bar = ProgressBar(max_value=len(ds.train_ds))
    i = 0
    loss = 0
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    print(ds[1][0].tweets[1])
    for user, label in train_loader:
        i += 1
        user_id += 1
        cf_loss, prediction = model.update_buffer(user[0], label,
                                                  need_backward=False,
                                                  train_classifier=False,
                                                  update_buffer=False, index=user_id)
        if prediction == label[0]:
            correct += 1
            if prediction == 1:
                tp += 1
            else:
                tn += 1
        else:
            if prediction == 1:
                fp += 1
            else:
                fn += 1
        bar.update(i)
    train_accuracy = correct / len(ds.train_ds)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    n_precision = tn / (tn + fn + eps)
    n_recall = tn / (tn + fp + eps)
    macro_P = (precision + n_precision) / 2
    macro_R = (recall + n_recall) / 2
    F1 = 2 * precision * recall / (precision + recall + eps)
    n_F1 = 2 * n_precision * n_recall / (n_precision + n_recall + eps)
    macro_F1 = (F1 + n_F1) / 2
    bar.finish()
    print(
        f'Train: Acc {train_accuracy}\tprecision {macro_P}\trecall {macro_R}\tF1 {macro_F1}\tloss {loss}')

    loss = 0
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for user, label in dev_loader:
            user_id += 1
            cf_loss, prediction = model.update_buffer(user[0], label,
                                                      need_backward=False,
                                                      train_classifier=False,
                                                      update_buffer=False)
            loss += cf_loss
            if prediction == label[0]:
                correct += 1
                if prediction == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if prediction == 1:
                    fp += 1
                else:
                    fn += 1
    dev_accuracy = correct / len(ds.dev_ds)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    n_precision = tn / (tn + fn + eps)
    n_recall = tn / (tn + fp + eps)
    macro_P = (precision + n_precision) / 2
    macro_R = (recall + n_recall) / 2
    F1 = 2 * precision * recall / (precision + recall + eps)
    n_F1 = 2 * n_precision * n_recall / (n_precision + n_recall + eps)
    macro_F1 = (F1 + n_F1) / 2
    print(
        f'Dev: Acc {dev_accuracy}\tprecision {macro_P}\trecall {macro_R}\tF1 {macro_F1}\tloss {loss}')

if __name__ == '__main__':
    inference()