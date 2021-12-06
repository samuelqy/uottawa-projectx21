# import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as dataset
# from PIL import Image
# vgg = torchvision.models.vgg16(pretrained=True)

# train_dir = '../DepressionMultiModel'
# train_dataset = dataset.ImageFolder(train_dir,
#                                     transforms.Compose(
#                                         [transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#                                         transforms.Normalize(mean=(0.5, 0.5,0.5), std=(0.5, 0.5, 0.5))
#                                         ]
#                                     ))
#
# for i in train_dataset:
#     print(i[0].size())
#     print(vgg(i[0].unsqueeze(0)).size())
# transform = transforms.Compose([transforms.CenterCrop(224)])
# image = Image.open(open('image/image.png', 'rb'))
# x, y = image.size
# if x > y:
#     x = 224 * x // y
#     y = 224
# else :
#     x = 224
#     y = 224 * x // y
# image = image.resize((x, y), Image.ANTIALIAS)
# # image.show()
# image = transform(image)
# # image.show()
# print(type(image))
# import os
# print(os.path.getsize('1.jpg'))

import os, json, random

# filtered = set()
# origin = os.listdir('negative')
# tofiltered = os.listdir('mm_n')
# # youchongfude = os.listdir('../mm_p')
# for o in origin:
#     i = 0
#     for p in tofiltered:
#         if o in p:
#             i += 1
#     if i > 1:
#         filtered.add(o)
#     if i == 0:
#         raise Exception('error')
#
# print(filtered)
# for f in filtered:
#     for to in tofiltered:
#         if f in to:
#             os.remove(f'mm_n/{to}')
#             print(f'remove mm_n/{to}')
#     for o in origin:
#         if f in o:
#             os.system(f'rm -fr negative/{o}')
#             print(f'remove negative/{o}')
# items = os.listdir('positive')
# for i, item in enumerate(items):
#     if (i+1) <= 1402:
#         os.system(f'cp positive/{item} dataset/train/positive -fr')
#     elif (i+1) >= 1403 and (i+1) <= 1856:
#         os.system(f'cp positive/{item} dataset/test/positive -fr')
#     else:
#         print(i)

def cal_metrics(prediction, ground_truth):
    eps = 1e-15
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    total_num = len(ground_truth)

    for (p, l) in zip(prediction, ground_truth):
        if p == l:
            correct += 1
            if p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1
    acc = correct / total_num
    precision = tp / (tp + fp + eps)
    n_precision = tn / (tn + fn + eps)
    recall = tp / (tp + fn + eps)
    n_recall = tn / (tn + fp + eps)
    F1 = 2 * precision * recall / (precision + recall + eps)
    n_F1 = 2 * n_precision * n_recall / (n_precision + n_recall +eps)
    m_precision = (precision + n_precision) / 2
    m_recall = (recall + n_recall) / 2
    m_F1 = (F1 + n_F1) / 2
    return acc, m_precision, m_recall, m_F1
import json
from string import ascii_letters, punctuation, printable
from dataHelper import isEmoji
import langid
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.casual import TweetTokenizer
stemmer = LancasterStemmer()
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
translate_table = dict((ord(char), None) for char in punctuation)

def clean_str(text):
    cleaned_tweet = [w.translate(translate_table) for w in tokenizer.tokenize(text) if w[:4] != 'http' and w not in punctuation and not isEmoji(w)]
    return ' '.join(cleaned_tweet)

keep_char = printable + '，。、【】；…’—♥”"'
def clear_ds():
    i = 0
    k = 0
    for user in os.listdir('dataset/train/positive'):
        delete = False
        k += 1
        print(k)
        with open(f'dataset/train/positive/{user}/timeline.json', encoding='utf-8') as f:
            r = 0
            lines = f.readlines()
            total = len(lines)
            for line in lines:
            # f.readline()
            # line = f.readline()
                j = json.loads(line, encoding='utf-8')
                text = j['text']
                text = clean_str(text)
                if langid.classify(text)[0] == 'ar' or langid.classify(text)[0] == 'ru' or langid.classify(text)[0] == 'ja':
                    r += 1
        if r / total > 0.1:
            delete = True
        if delete:
            i += 1
            # os.system(f'rm -fr dataset/test/positive/{user}')
    print(i)

def make_new_ds():
    positive_users = eval(open('../positive_users.txt').read())
    negative_users = eval(open('../negative_users.txt').read())
    positive_timelines = os.listdir('mmds/positive')
    negative_timelines = os.listdir('mmds/negative')
    for user in positive_users:
        for timeline in positive_timelines:
            if user in timeline:
                os.system(f'cp mmds/positive/{timeline} new_ds/positive/{user}/timeline.txt')
    for user in negative_users:
        for timeline in negative_timelines:
            if user in timeline:
                os.system(f'cp mmds/negative/{timeline} new_ds/negative/{user}/timeline.txt')

import pickle
import numpy as np
def random_half():
    roots = ['positive', 'negative']
    h_roots = ['half_p', 'half_n']
    for root, h_root in zip(roots, h_roots):
        users = os.listdir(f'new_ds/{root}')
        for user in users:
            with open(f'new_ds/{root}/{user}/tweets.pkl', 'rb') as f:
                tweets = pickle.load(f)
                half_tweets = []
                image_num = tweets[0]
                del tweets[0]
                index = np.random.choice(len(tweets), len(tweets)//2, False)
                index.sort()
                for i in index:
                    half_tweets.append(tweets[i])
                half_tweets.insert(0, image_num)
                os.system(f'mkdir new_ds/{h_root}/{user}')
                os.system(f'cp new_ds/{root}/{user}/vocab.pkl new_ds/{h_root}/{user}/vocab.pkl')
                with open(f'new_ds/{h_root}/{user}/tweets.pkl', 'wb') as h_f:
                    pickle.dump(half_tweets, h_f)
        print(f'Process {root} over!')

if __name__ == '__main__':
    random_half()