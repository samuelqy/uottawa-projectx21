from model import MultiAgentClassifier, TextClassifier, ImageClassifier, ConcatClassifier, DAN, CAN, MAN
import torch
import json
from torch.utils.data import DataLoader
from dataHelper import collate, get_type_dataset
from progressbar import ProgressBar
import torch.optim as optim
import torch.nn as nn
import argparse
from itertools import count
from utils import cal_metrics
import os

parser = argparse.ArgumentParser()
parser.add_argument('--type')
parser.add_argument('--update_scale')
parser.add_argument('--ds')
parser.add_argument('--half')
parser.add_argument('--save_model')
parser.add_argument('--save_path')
args = parser.parse_args()
model_type = args.type
update_scale = float(args.update_scale)
ds = args.ds
half = True if args.half == 'true' else False
save_model = True if args.save_model == 'true' else False
save_path = args.save_path

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(1)
torch.cuda.manual_seed(1)
eps = 1e-15

def evaluate_rl():
    global ds
    ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=True, ds=ds, half=half)


    test = []
    train = []



    positive_train_user = []
    positive_count = 0
    for user in ds.train_ds:
        if user[1] == 1:
            positive_train_user.append(user)
            positive_count += 1

    for i in range(len(ds.train_ds)//positive_count):
        for user in positive_train_user:
            ds.train_ds.add(user)


    for user in ds.train_ds:
        train.append(user[2])
    for user in ds.dev_ds:
        test.append(user[2])

    file = open("test_users.txt", "w")
    for user in test:
        file.write(user + "\n")
    file.close()

    file2 = open("train_users.txt", "w")
    for user in train:
        file2.write(user + "\n")
    file2.close()


    pre_train_loader = DataLoader(ds.train_ds, batch_size=1, collate_fn=collate, shuffle=True)
    pre_dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)
    train_loader = DataLoader(ds.train_ds, batch_size=500, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)


    print(len(ds.train_ds), len(ds.dev_ds))
    if os.path.exists('models/baseModelGRU.pt'):
        model = torch.load('models/baseModelGRU.pt')
        model.textEncoder.encoder.flatten_parameters()
    else:
        model = MultiAgentClassifier(len(ds.vocab), ds.word2id)
        # pre-train classifier
        pre_train_max_F1 = 0
        d_epochs = 0
        for epoch in count(1):
            print(f'Epoch {epoch}')
            bar = ProgressBar(max_value=len(ds.train_ds))
            i = 0
            loss = 0
            correct = 0
            tp, tn, fp, fn = 0, 0, 0, 0
            for user, label in pre_train_loader:
                i += 1
                bar.update(i)
                cf_loss, prediction = model.pre_train_classifier(user[0], label)
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
            print(f'Pre-train: Acc {train_accuracy}\tprecision {macro_P}\trecall {macro_R}\tF1 {macro_F1}\tloss {loss}')

            loss = 0
            correct = 0
            tp, tn, fp, fn = 0, 0, 0, 0
            with torch.no_grad():
                for user, label in pre_dev_loader:
                    cf_loss, prediction = model.pre_train_classifier(user[0], label, need_backward=False)
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
            print(f'Pre-dev: Acc {dev_accuracy}\tprecision {macro_P}\trecall {macro_R}\tF1 {macro_F1}\tloss {loss}\t')
            if pre_train_max_F1 < macro_F1:
                pre_train_max_F1 = macro_F1
                if save_model:
                    torch.save(model, 'models/baseModelGRU.pt')
                    print('Save successfully!')
                #for efficiency, need to comment out for higher f1 score
                d_epochs = 0
                d_epochs += 1
            else:
                d_epochs += 1
            if d_epochs == 4:
                break
            if macro_F1 > 0.825:
                break

    # joint training
    max_F1 = 0
    joint_epoch = 0
    for epoch in count(1):
        joint_epoch += 1
        print(f'Epoch {epoch}')
        bar = ProgressBar(max_value=len(ds.train_ds))
        i = 0
        loss = 0
        correct = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for user, label in train_loader:
            i += 1
            cf_loss, prediction = model.joint_train(user[0], label)
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
        validation_dict = {}

        with torch.no_grad():
            for user, label in dev_loader:
                id = user[2]
                cf_loss, prediction, prob = model.update_buffer(user[0], label,
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

                validation_dict[id] = prob

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


        with open('test_probabilities.txt', 'w') as file:
            file.write(json.dumps(validation_dict))

        
        print(
            f'Dev: Acc {dev_accuracy}\tprecision {macro_P}\trecall {macro_R}\tF1 {macro_F1}\tloss {loss}')
        if save_model:
            if max_F1 < macro_F1:
                max_F1 = macro_F1
                torch.save(model, save_path)
                print('Save successfully!')

        #for efficiency i set the limitation of number of epochs, in original source code, there is no limitation
        if (joint_epoch == 10):
            break


def evaluate_text():
    global ds
    ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=False, ds=ds, half=half)
    train_loader = DataLoader(ds.train_ds, batch_size=1, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)
    model = TextClassifier(vocab_size=len(ds.vocab), embedding_size=64, word2id=ds.word2id)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    print(len(ds.train_ds), len(ds.dev_ds))
    max_F1 = 0
    for epoch in count(1):
        print(f'Epoch {epoch}')
        # Train step
        bar = ProgressBar(max_value=len(ds.train_ds))
        i = 0
        predictions = []
        ground_truth = []
        train_loss = 0
        for user, label in train_loader:
            i += 1
            user = user[0]
            ground_truth += label
            output = model(user)
            prediction = torch.max(output, 1, keepdim=False)[1].item()
            predictions.append(prediction)
            optimizer.zero_grad()
            loss = loss_fn(output, torch.Tensor(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.update(i)
        acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
        bar.finish()
        print(f'Train acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {train_loss}')

        # Dev step
        predictions = []
        ground_truth = []
        dev_loss = 0
        with torch.no_grad():
            for user, label in dev_loader:
                user = user[0]
                ground_truth += label
                output = model(user)
                prediction = torch.max(output, 1, keepdim=False)[1].item()
                predictions.append(prediction)
                loss = loss_fn(output, torch.Tensor(label).long().to(device))
                dev_loss += loss.item()
            acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
            print(f'Dev acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {dev_loss}')
            if save_model:
                if F1 > max_F1:
                    max_F1 = F1
                    torch.save(model, save_path)


def evaluate_image():
    global ds
    ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=True, ds=ds, half=half)
    train_loader = DataLoader(ds.train_ds, batch_size=1, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)
    model = ImageClassifier()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    print(len(ds.train_ds), len(ds.dev_ds))
    max_F1 = 0
    for epoch in count(1):
        print(f'Epoch {epoch}')
        # Train step
        bar = ProgressBar(max_value=len(ds.train_ds))
        i = 0
        predictions = []
        ground_truth = []
        train_loss = 0
        for user, label in train_loader:
            i += 1
            user = user[0]
            ground_truth += label
            output = model(user)
            prediction = torch.max(output, 1, keepdim=False)[1].item()
            predictions.append(prediction)
            optimizer.zero_grad()
            loss = loss_fn(output, torch.Tensor(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.update(i)
        acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
        bar.finish()
        print(f'Train acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {train_loss}')

        # Dev step
        predictions = []
        ground_truth = []
        dev_loss = 0
        with torch.no_grad():
            for user, label in dev_loader:
                user = user[0]
                ground_truth += label
                output = model(user)
                prediction = torch.max(output, 1, keepdim=False)[1].item()
                predictions.append(prediction)
                loss = loss_fn(output, torch.Tensor(label).long().to(device))
                dev_loss += loss.item()
            acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
            print(f'Dev acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {dev_loss}')
            if save_model:
                if F1 > max_F1:
                    max_F1 = F1
                    torch.save(model, save_path)


def evaluate_concat():
    global ds
    ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=True, ds=ds, half=half)
    train_loader = DataLoader(ds.train_ds, batch_size=1, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)
    model = ConcatClassifier(vocab_size=len(ds.vocab), word2id=ds.word2id)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    print(len(ds.train_ds), len(ds.dev_ds))
    for epoch in count(1):
        print(f'Epoch {epoch}')
        # Train step
        bar = ProgressBar(max_value=len(ds.train_ds))
        i = 0
        predictions = []
        ground_truth = []
        train_loss = 0
        for user, label in train_loader:
            i += 1
            ground_truth += label
            output = model(user)
            prediction = torch.max(output, 1, keepdim=False)[1].item()
            predictions.append(prediction)
            optimizer.zero_grad()
            loss = loss_fn(output, torch.Tensor(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.update(i)
        acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
        bar.finish()
        print(f'Train acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {train_loss}')

        # Dev step
        predictions = []
        ground_truth = []
        dev_loss = 0
        with torch.no_grad():
            for user, label in dev_loader:
                ground_truth += label
                output = model(user)
                prediction = torch.max(output, 1, keepdim=False)[1].item()
                predictions.append(prediction)
                loss = loss_fn(output, torch.Tensor(label).long().to(device))
                dev_loss += loss.item()
            acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
            print(f'Dev acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {dev_loss}')


def evaluate_DAN():
    global ds
    ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=True, ds=ds, half=half)
    train_loader = DataLoader(ds.train_ds, batch_size=1, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)
    print(len(ds.train_ds), len(ds.dev_ds))
    model = DAN(len(ds.vocab), embedding_size=100, word2id=ds.word2id, fix_length=49)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    max_F1 = 0
    for epoch in count(1):
        print(f'Epoch {epoch}')
        bar = ProgressBar(max_value=len(ds.train_ds))
        i = 0
        predictions = []
        ground_truth = []
        train_loss = 0
        for user, label in train_loader:
            i += 1
            output = model(user)
            ground_truth += label
            prediction = torch.max(output, 1, keepdim=False)[1].item()
            predictions.append(prediction)
            optimizer.zero_grad()
            loss = loss_fn(output, torch.Tensor(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.update(i)
        acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
        bar.finish()
        print(f'Train acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {train_loss}')

        # Dev step
        predictions = []
        ground_truth = []
        dev_loss = 0
        with torch.no_grad():
            for user, label in dev_loader:
                ground_truth += label
                output = model(user)
                prediction = torch.max(output, 1, keepdim=False)[1].item()
                predictions.append(prediction)
                loss = loss_fn(output, torch.Tensor(label).long().to(device))
                dev_loss += loss.item()
            acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
            print(f'Dev acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {dev_loss}')


def evaluate_CAN():
    global ds
    ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=True, ds=ds, half=half)
    train_loader = DataLoader(ds.train_ds, batch_size=1, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)
    print(len(ds.train_ds), len(ds.dev_ds))
    model = CAN(len(ds.vocab), embedding_size=100, word2id=ds.word2id, fix_length=49)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    max_F1 = 0
    for epoch in count(1):
        print(f'Epoch {epoch}')
        bar = ProgressBar(max_value=len(ds.train_ds))
        i = 0
        predictions = []
        ground_truth = []
        train_loss = 0
        for user, label in train_loader:
            i += 1
            output = model(user)
            ground_truth += label
            prediction = torch.max(output, 1, keepdim=False)[1].item()
            predictions.append(prediction)
            optimizer.zero_grad()
            loss = loss_fn(output, torch.Tensor(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.update(i)
        acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
        bar.finish()
        print(f'Train acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {train_loss}')

        # Dev step
        predictions = []
        ground_truth = []
        dev_loss = 0
        with torch.no_grad():
            for user, label in dev_loader:
                ground_truth += label
                output = model(user)
                prediction = torch.max(output, 1, keepdim=False)[1].item()
                predictions.append(prediction)
                loss = loss_fn(output, torch.Tensor(label).long().to(device))
                dev_loss += loss.item()
            acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
            print(f'Dev acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {dev_loss}')


def evaluate_MAN():
    global ds
    ds = get_type_dataset(load_pickle=True, build_pickle=False, need_image=True, ds=ds, half=half)
    train_loader = DataLoader(ds.train_ds, batch_size=1, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ds.dev_ds, batch_size=1, collate_fn=collate)
    print(len(ds.train_ds), len(ds.dev_ds))
    model = MAN(len(ds.vocab), embedding_size=100, word2id=ds.word2id)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    max_F1 = 0
    for epoch in count(1):
        print(f'Epoch {epoch}')
        bar = ProgressBar(max_value=len(ds.train_ds))
        i = 0
        predictions = []
        ground_truth = []
        train_loss = 0
        for user, label in train_loader:
            i += 1
            output = model(user)
            ground_truth += label
            prediction = torch.max(output, 1, keepdim=False)[1].item()
            predictions.append(prediction)
            optimizer.zero_grad()
            loss = loss_fn(output, torch.Tensor(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.update(i)
        acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
        bar.finish()
        print(f'Train acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {train_loss}')

        # Dev step
        predictions = []
        ground_truth = []
        dev_loss = 0
        with torch.no_grad():
            for user, label in dev_loader:
                ground_truth += label
                output = model(user)
                prediction = torch.max(output, 1, keepdim=False)[1].item()
                predictions.append(prediction)
                loss = loss_fn(output, torch.Tensor(label).long().to(device))
                dev_loss += loss.item()
            acc, precision, recall, F1 = cal_metrics(predictions, ground_truth)
            print(f'Dev acc {acc}\tprecision {precision}\trecall {recall}\tF1 {F1}\tloss {dev_loss}')

if __name__ == '__main__':
    if model_type == 'text':
        evaluate_text()
    elif model_type == 'image':
        evaluate_image()
    elif model_type == 'concat':
        evaluate_concat()
    elif model_type == 'multi-agent':
        evaluate_rl()
    elif model_type == 'DAN':
        evaluate_DAN()
    elif model_type == 'CAN':
        evaluate_CAN()
    elif model_type == 'MAN':
        evaluate_MAN()