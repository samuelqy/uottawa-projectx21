from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torchvision import transforms
from PIL import Image
import os
from progressbar import ProgressBar
from PIL import ImageFile
from torchvision.models import vgg16
import torch
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from multiprocessing import Pool
from string import punctuation
import pickle
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.cuda
import json

use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(1)
torch.cuda.manual_seed(1)

english_stopwords = stopwords.words('english')
stemmer = LancasterStemmer()
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
translate_table = dict((ord(char), None) for char in punctuation)

def isEmoji(content):
    if not content:
        return False
    if u"\U0001F600" <= content and content <= u"\U0001F64F":
        return True
    elif u"\U0001F300" <= content and content <= u"\U0001F5FF":
        return True
    elif u"\U0001F680" <= content and content <= u"\U0001F6FF":
        return True
    elif u"\U0001F1E0" <= content and content <= u"\U0001F1FF":
        return True
    else:
        return False

class ImageDataset(Dataset):
    def __init__(self, root, need_split=False, train_ratio=None):
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # 此处需修改PyTorch内置的vgg代码，返回的是全连接层的上一层输出的特征
        self.vgg = vgg16(pretrained=True)
        self.vgg.to(device)
        self.vgg.eval()
        self.ds = []
        self.initializer(root)
        self.train_ds, self.test_ds = None, None
        if need_split:
            if train_ratio == None:
                raise Exception('Error: train_ration cannot be None if need_split is True!')
            self.train_ds, self.dev_ds = self.train_test_split(train_ratio)

    def initializer(self, root):
        # load train and dev dataset
        categories = ['positive', 'negative']
        for c in categories:
            users = [os.path.join(root, c, user) for user in os.listdir(os.path.join(root, c)) if user[0] != '.']
            bar = ProgressBar(max_value=len(users)).start()
            i = 0
            for user in users:
                i += 1
                bar.update(i)
                images = [self.imageHandler(os.path.join(user, image_path)) for image_path in os.listdir(user) if
                          image_path != 'timeline.json' and image_path[-4:] not in ['.txt' , '.pkl']]
                # timeline_path = os.path.join(root, c, user, 'timeline.json')
                # label = 1 if c == 'positive' else 0
                # self.ds.append([images, timeline_path, label])
            bar.finish()

    def train_test_split(self, train_ratio):
        size = len(self.ds)
        return random_split(self, [int(size * train_ratio), size - int(size * train_ratio)])

    def imageHandler(self, image_path: str) -> Image:
        if os.path.getsize(image_path) == 0:
            os.remove(image_path)
            return None
        image = Image.open(open(image_path, 'rb'))
        image = image.convert('RGB')  # 将所有图片转换成3通道
        x, y = image.size
        if x > y:
            x_ = 224 * x // y
            y_ = 224
        else:
            x_ = 224
            y_ = 224 * y // x
        image = image.resize((x_, y_), Image.ANTIALIAS)
        image = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            try:
                image_feature = self.vgg(image)
            except Exception:
                print(image_path)
            with open(image_path[:-4] + '.txt', 'w') as f:
                # print(f.name, image_path[:-4]+'.pkl')
                f.write(str(image_feature.tolist()))
        return image  # (3, 224, 224) -> (1, 3, 224, 224)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        return self.ds[item]

class User:
    def __init__(self, user_path, load_pickle, build_pickle, need_image):
        self.tweets = []
        self.vocab = set()
        self.image_num = 0
        self.build_pickle = build_pickle
        self.need_image = need_image
        if load_pickle:
            self.load(user_path)
        else:
            self.process(user_path)
            if build_pickle:
                self.dump(user_path)

    def process(self, user_path):
        file_names = os.listdir(user_path)
        if self.need_image:
            ids_with_image = [file_name[:-4] for file_name in file_names if file_name != 'timeline.json' and file_name[-4:] == '.txt']
        with open(os.path.join(user_path, 'timeline.txt'),encoding='gbk',errors = 'ignore') as f:
            for line in f.readlines():
                j = json.loads(line, encoding='utf-8')
                tweet_id = j['id_str']
                text = j['text']
                if 'diagnosed' in text:
                    continue
                if self.need_image:
                    if tweet_id in ids_with_image:
                        with open(os.path.join(user_path, tweet_id + '.txt')) as v:
                            image_feature = torch.Tensor(eval(v.read())).view(1, -1).float()
                        self.image_num += 1
                    else:
                        image_feature = torch.zeros(1, 7 * 7 * 512)
                else:
                    image_feature = None
                    self.image_num += 1
                cleaned_tweet = [w.lower().translate(translate_table) for w in tokenizer.tokenize(text) if w[:4] != 'http' and w not in punctuation]
                if len(cleaned_tweet) == 0:
                    continue
                self.tweets.append((cleaned_tweet, image_feature, tweet_id))
                self.vocab.update(cleaned_tweet)
        if self.build_pickle:
            self.tweets.insert(0, self.image_num)

    def dump(self, user_path):
        with open(os.path.join(user_path, 'tweets_with_id.pkl'), 'wb') as f:
            pickle.dump(self.tweets, f)
        with open(os.path.join(user_path, 'vocab.pkl'), 'wb') as f:
            pickle.dump(self.vocab, f)

    def load(self, user_path):
        with open(os.path.join(user_path, 'tweets_with_id.pkl'), 'rb') as f:
            self.tweets = pickle.load(f)
            self.image_num = self.tweets[0]
            del self.tweets[0]
        with open(os.path.join(user_path, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)

    def __len__(self):
        return len(self.tweets)

class UserDataset(Dataset):
    def __init__(self, root, need_split=True, train_ratio=None, load_pickle=True, build_pickle=True, need_image=True, half=False):
        self.ds = []
        self.vocab = set()
        self.word2id = {}
        self.need_image = need_image
        self.initializer(root, load_pickle, build_pickle, half)
        if need_split:
            if train_ratio is None:
                raise Exception('Error: train_ration cannot be None if need_split is True!')
            self.train_ds, self.dev_ds = self.train_dev_split(train_ratio)

    def initializer(self, root, load_pickle, build_pickle, half):
        if half:
            categories = ['half_p', 'half_n']

        elif root == 'projectx' :

            categories = ['positive', 'negative']

        else:
            categories = ['positive', 'negative']
        for c in categories:
            users = [os.path.join(root, c, user) for user in os.listdir(os.path.join(root, c)) if user[0] != '.']
            bar = ProgressBar(max_value=len(users)).start()
            i = 0
            for user_path in users:
                
                i += 1
                bar.update(i)

                #there exist some users triger error while encoding their timeline.txt, might because their timeline contains foreign languages
                try:
                    user_data = User(user_path, load_pickle=load_pickle, build_pickle=build_pickle, need_image=self.need_image)
                except BaseException:
                    print(user_path)
                    shutil.rmtree(user_path)
                    continue



                if len(user_data) < 2:
                    continue
                self.vocab.update(user_data.vocab)
                del user_data.vocab
                label = 1 if c == 'positive' else 0
                # self.ds.append((user_data, label, user_path.split('/')[-1]))
                self.ds.append((user_data, label))
            bar.finish()
        self.word2id = {word:id for id, word in enumerate(self.vocab)}

    def train_dev_split(self, train_ratio):
        size = len(self.ds)
        return random_split(self, [int(size * train_ratio), size - int(size * train_ratio)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        return self.ds[item]

def get_type_dataset(load_pickle, build_pickle, need_image, ds, half):
    if ds == 'new_ds':
        if half:
            return UserDataset('new_ds', need_split=True,
                               train_ratio=0.8,
                               load_pickle=load_pickle,
                               build_pickle=build_pickle,
                               need_image=need_image,
                               half=half)
        else:
            

            return UserDataset('new_ds', need_split=True,
                               train_ratio=0.8,
                               load_pickle=load_pickle,
                               build_pickle=build_pickle,
                               need_image=need_image,
                               half=half)
    elif ds == 'selected':
        return  _UserDataset('new_ds/selected_pair')
    elif ds == 'unselected':
        return _UserDataset('new_ds/unselected_pair')
    elif ds == 'old':
        return UserDataset('dataset/train', need_split=True,
                           train_ratio=0.8,
                           load_pickle=load_pickle,
                           build_pickle=build_pickle,
                           need_image=need_image)
    elif ds == 'projectx':
        return UserDataset('projectx', need_split=True,
                               train_ratio=0.8,
                               load_pickle=load_pickle,
                               build_pickle=build_pickle,
                               need_image=need_image,
                               half=half)

    else:
        return UserDataset(f'dataset/{ds}', need_split=True,
                           train_ratio=0.8,
                           load_pickle=True,
                           build_pickle=False,
                           need_image=True,
                           half=False)
    # else:   # type == 'test'
    #     return UserDataset('dataset/test', need_split=True, train_ratio=0.8, load_pickle=load_pickle, build_pickle=build_pickle, need_image=need_image)















# the code below seems was abandoned by author, their functionality was achieved by code above.

class _User:
    def __init__(self, user_path):
        self.tweets = []
        self.image_num = 0
        self.load(user_path)

    def dump(self, user_path):
        with open(os.path.join(user_path, 'tweets_with_id.pkl'), 'wb') as f:
            pickle.dump(self.tweets, f)
        # with open(os.path.join(user_path, 'vocab.pkl'), 'wb') as f:
        #     pickle.dump(self.vocab, f)

    def load(self, user_path):
        with open(user_path, 'rb') as f:
            self.tweets = pickle.load(f)
            self.image_num = self.tweets[0]
            del self.tweets[0]

    def __len__(self):
        return len(self.tweets)

class _UserDataset(Dataset):
    def __init__(self, root, need_split=True, train_ratio=0.8):
        self.ds = []
        self.vocab = set()
        self.word2id = {}
        self.initializer(root)
        if need_split:
            if train_ratio is None:
                raise Exception('Error: train_ration cannot be None if need_split is True!')
            self.train_ds, self.dev_ds = self.train_dev_split(train_ratio)

    def initializer(self, root):
        categories = ['positive', 'negative']
        for c in categories:
            users = os.listdir(f'new_ds/{c}')
            for user in users:
                with open(f'new_ds/{c}/{user}/vocab.pkl', 'rb') as f:
                    _vocab = pickle.load(f)
                    self.vocab.update(_vocab)
            users = os.listdir(os.path.join(root, c))
            bar = ProgressBar(max_value=len(users)).start()
            i = 0
            for user_path in users:
                i += 1
                bar.update(i)
                user_data = _User(os.path.join(root, c, user_path))
                if len(user_data) < 2:
                    continue
                label = 1 if c == 'positive' else 0
                # self.ds.append((user_data, label, user_path.split('/')[-1]))
                self.ds.append((user_data, label))
            bar.finish()

        self.word2id = {word:id for id, word in enumerate(self.vocab)}

    def train_dev_split(self, train_ratio):
        size = len(self.ds)
        return random_split(self, [int(size * train_ratio), size - int(size * train_ratio)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        return self.ds[item]

def get_dataset():
    pool = Pool(processes=2)
    jobs = [pool.apply_async(get_type_dataset, ('train', )), pool.apply_async(get_type_dataset, ('test', ))]
    datasets = [j.get() for j in jobs]
    return datasets[0].train_ds, datasets[0].dev_ds, datasets[1]

def collate(batch_data):
    batch_size = len(batch_data)
    users = []
    labels = []
    for i in range(batch_size):
        users.append(batch_data[i][0])
        labels.append(batch_data[i][1])
    return users, labels




#main function, used to build pickle file of dataset, 
if __name__ == '__main__':
    
    # build the .pkl files for projectx dataset
    # i added this line to build image feature .txt file before build .pkl file to ensure the .pkl file also containsimage feature.
    image_feature = ImageDataset('projectx')
    get_type_dataset(load_pickle=False, build_pickle=True, need_image=True, ds='projectx', half=False)

    
    # if want to build .txt and .pkl for original dataset, change to
    #image_feature = ImageDataset('new_ds')
    #get_type_dataset(load_pickle=False, build_pickle=True, need_image=True, ds='new_ds', half=False)