import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions import Categorical
from collections import deque
import random
import torch.optim as optim
import pickle
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(1)
torch.cuda.manual_seed(1)

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, user):
        images = []
        for _, v in user.tweets:
            if torch.sum(v).item() != 0:
                images.append(v.to(device))
        if len(images) == 0:
            feat = torch.zeros(1, 512 * 7 * 7).to(device)
        else:
            images = torch.cat(images, dim=0)
            feat = torch.mean(images, dim=0, keepdim=True)
        prediction = self.classifier(feat)
        return prediction


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, word2id):
        super(TextClassifier, self).__init__()
        self.embedding_size = embedding_size
        self.word2id = word2id
        self.embeddingLayer = nn.Embedding(vocab_size+2, embedding_size, padding_idx=vocab_size+1)
        self.sentenceEncoder = nn.GRU(input_size=embedding_size, hidden_size=embedding_size//2, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 2)
        )

    def forward(self, user):
        text = [item[0] for item in user.tweets]
        max_length = max(len(s) for s in text)
        embedding, sorted_len, reversed_indices = self.embedding_lookup(text, max_length)
        packed_embed = pack_padded_sequence(embedding, sorted_len, batch_first=True)
        _, h = self.sentenceEncoder(packed_embed)
        h = h[-2:]
        h = torch.cat([h[0], h[1]], dim=-1)
        feat = h[reversed_indices]
        user_feat = torch.mean(feat, dim=0, keepdim=True)
        prediction = self.classifier(user_feat)
        return prediction

    def embedding_lookup(self, sentences, max_length):
        ids = []
        lengths = []
        for sentence in sentences:
            id = []
            lengths.append(len(sentence))
            for word in sentence:
                if word in self.word2id:
                    id.append(self.word2id[word])
                else:
                    id.append(self.embeddingLayer.padding_idx - 1)
            id += [self.embeddingLayer.padding_idx for _ in range(max_length - len(id))]
            ids.append(id)
        ids = torch.LongTensor(ids).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        sorted_len, indices = torch.sort(lengths, 0, descending=True)
        _, reversed_indices = torch.sort(indices, 0)
        ids = ids[indices]
        return self.embeddingLayer(ids), sorted_len.tolist(), reversed_indices.to(device)


class ImageRLClassifier(nn.Module):
    def __init__(self):
        super(ImageRLClassifier, self).__init__()
        self.selected_data = []
        self.unselected_data = []
        self.saved_log_probs = []
        self.env = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=-1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, user):
        images = [image for _, image in user.tweets]
        while len(self.selected_data) == 0:
            for image in images:
                state_A = torch.mean(torch.cat(self.selected_data, dim=0), dim=0, keepdim=True) if len(self.selected_data) == 0 else torch.zeros(1, 512 * 7 * 7).to(device)
                state_B = torch.mean(torch.cat(self.unselected_data, dim=0), dim=0, keepdim=True) if len(self.unselected_data) == 0 else torch.zeros(1, 512 * 7 * 7).to(device)
                state_C = image.to(device)
                state = torch.cat([state_A, state_B, state_C], dim=-1)
                probs = self.env(state)
                sampler = Categorical(probs)
                action = sampler.sample()
                if action.item() == 1:
                    self.selected_data.append(state_C)
                else:
                    self.unselected_data.append(state_C)


class ConcatClassifier(nn.Module):
    def __init__(self, vocab_size, word2id):
        super(ConcatClassifier, self).__init__()
        self.textEncoder = TextEncoder(input_dim=128,
                                       encoding_dim=128,
                                       vocab_size=vocab_size,
                                       word2id=word2id)
        self.imageEncoder = ImageEncoder(input_dim=7 * 7 * 512,
                                     encoding_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, user):
        user = user[0]
        features = []
        for text, image in user.tweets:
            text_feature = self.textEncoder([text])
            image_feature = self.imageEncoder(image.to(device))
            feature = torch.cat([text_feature, image_feature], dim=-1)
            features.append(feature)
        mean_feature = torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)
        output = self.classifier(mean_feature)
        return output


class Buffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def append(self, text_s, image_s, text_a, image_a, r, text_next_s, image_next_s, opposed_text_a, opposed_image_a):
        transition = (text_s, image_s, text_a, image_a, r, text_next_s, image_next_s, opposed_text_a, opposed_image_a)
        self.buffer.append(transition)

    def sample(self, num):
        num = min(num, len(self.buffer))
        batch = random.sample(self.buffer, num)
        batch_text_s = [t[0] for t in batch]
        batch_image_s = [t[1] for t in batch]
        batch_text_a = [t[2] for t in batch]
        batch_image_a = [t[3] for t in batch]
        batch_r = [t[4] for t in batch]
        batch_text_next_s = [t[5] for t in batch]
        batch_image_next_s = [t[6] for t in batch]
        batch_opposed_text_a = [t[7] for t in batch]
        batch_opposed_image_a = [t[8] for t in batch]

        batch_text_s = torch.cat(batch_text_s, dim=0)
        batch_image_s = torch.cat(batch_image_s, dim=0)
        batch_text_a = torch.cat(batch_text_a, dim=0)
        batch_image_a = torch.cat(batch_image_a, dim=0)
        batch_r = torch.Tensor(batch_r).unsqueeze(-1).to(device)
        batch_text_next_s = torch.cat(batch_text_next_s, dim=0)
        batch_image_next_s = torch.cat(batch_image_next_s, dim=0)
        batch_opposed_text_a = torch.cat(batch_opposed_text_a, dim=0)
        batch_opposed_image_a = torch.cat(batch_opposed_image_a, dim=0)

        return batch_text_s, batch_image_s, \
               batch_text_a, batch_image_a, \
               batch_r, \
               batch_text_next_s, batch_image_next_s, \
               batch_opposed_text_a, \
               batch_opposed_image_a

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]


class TextEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, vocab_size, word2id):
        super(TextEncoder, self).__init__()
        self.word2id = word2id
        self.embeddingLayer = nn.Embedding(vocab_size + 1, input_dim, padding_idx=vocab_size)
        # Because bidirectional is True, hidden_size needs to be encoding_dim // 2
        self.encoder = nn.GRU(input_size=input_dim,
                              hidden_size=encoding_dim // 2,
                              num_layers=1,
                              bidirectional=True,
                              batch_first=True)

    def forward(self, data):
        embeds, sorted_len, reversed_indices = self.embedding_lookup(data)
        packed_embeds = pack_padded_sequence(embeds, sorted_len, batch_first=True)
        _, sentence_feature = self.encoder(packed_embeds)
        sentence_feature = sentence_feature[-2:]
        encoding = torch.cat([sentence_feature[0], sentence_feature[1]], dim=-1)
        encoding = encoding[reversed_indices]
        return encoding

    def embedding_lookup(self, tweets):
        ids = []
        lengths = [len(tweet) for tweet in tweets]
        max_len = max(lengths)
        for tweet in tweets:
            id = []
            for word in tweet:
                if word in self.word2id:
                    id.append(self.word2id[word])
                else:
                    id.append(self.embeddingLayer.padding_idx - 1)
            id += [self.embeddingLayer.padding_idx] * (max_len - len(id))
            ids.append(id)
        ids = torch.Tensor(ids).long().to(device)
        lengths = torch.Tensor(lengths).long().to(device)
        sorted_len, indices = torch.sort(lengths, 0, descending=True)
        _, reversed_indices = torch.sort(indices, 0)
        ids = ids[indices]
        embeds = self.embeddingLayer(ids)
        return embeds, sorted_len.tolist(), reversed_indices.to(device)


class ImageEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(ImageEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,
                      encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim,
                      encoding_dim)
        )

    def forward(self, data):
        return self.encoder(data)


class BaseActor(nn.Module):
    def __init__(self, encoding_dim):
        super(BaseActor, self).__init__()
        self.encoding_dim = encoding_dim
        self.selected_data = [torch.zeros(1, self.encoding_dim).to(device)]
        self.unselected_data = [torch.zeros(1, self.encoding_dim).to(device)]

    def forward(self, data, need_backward, target):
        raise NotImplementedError

    def choose_action(self, s, need_backward, target):
        probs = self.map(s)
        if need_backward:
            sampler = Categorical(probs)
            action = sampler.sample()
        else:
            action = torch.max(probs, 1, keepdim=False)[1]
        if not target:
            return action.item()
        else:
            actions = action.unsqueeze(1)
            return torch.zeros(s.size(0), 2).to(device).scatter_(1, actions, 1)

    def get_log_probs(self, s, action):
        probs = self.map(s)
        sampler = Categorical(probs)
        log_probs = sampler.log_prob(action)
        return log_probs


class TextActor(BaseActor):
    def __init__(self, encoding_dim):
        super(TextActor, self).__init__(encoding_dim)
        self.map = nn.Sequential(
            nn.Linear(encoding_dim * 3,
                      encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim,
                      2),
            nn.Softmax(dim=-1)
        )

    def forward(self, tweet, need_backward, target):
        state_a = torch.mean(torch.cat(self.selected_data, dim=0), dim=0, keepdim=True)
        state_b = torch.mean(torch.cat(self.unselected_data, dim=0), dim=0, keepdim=True)
        state = torch.cat([state_a, state_b, tweet], dim=-1).detach()
        action = self.choose_action(state, need_backward, target)
        # target actor cannot update the data buffer
        if not target:
            if action == 1:
                self.selected_data.append(tweet)
            else:
                self.unselected_data.append(tweet)
        return state, action


class ImageActor(BaseActor):
    def __init__(self, encoding_dim):
        super(ImageActor, self).__init__(encoding_dim)
        self.map = nn.Sequential(
            nn.Linear(encoding_dim * 3,
                      encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim,
                      2),
            nn.Softmax(dim=-1)
        )

    def forward(self, image, need_backward, target):
        state_a = torch.mean(torch.cat(self.selected_data, dim=0), dim=0, keepdim=True)
        state_b = torch.mean(torch.cat(self.unselected_data, dim=0), dim=0, keepdim=True)
        state = torch.cat([state_a, state_b, image], dim=-1).detach()
        action = self.choose_action(state, need_backward, target)
        # target actor cannot update the data buffer
        if not target:
            if action == 1:
                self.selected_data.append(image)
            else:
                self.unselected_data.append(image)
        return state, action


class Critic(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, encoding_dim):
        super(Critic, self).__init__()
        self.QText = nn.Sequential(
            nn.Linear(text_input_dim + 2, text_input_dim),
            nn.ReLU(),
            nn.Linear(text_input_dim, encoding_dim),
            nn.Tanh()
        )
        self.QImage = nn.Sequential(
            nn.Linear(image_input_dim + 2, image_input_dim),
            nn.ReLU(),
            nn.Linear(image_input_dim, encoding_dim),
            nn.Tanh()
        )
        self.QScore = nn.Sequential(
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, 1)
        )

    def forward(self, text_s, text_a, image_s, image_a):
        text_rep = self.QText(torch.cat([text_s, text_a], dim=-1))
        image_rep = self.QImage(torch.cat([image_s, image_a], dim=-1))
        score = self.QScore(torch.cat([text_rep, image_rep], dim=-1))
        return score


class BCritic(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, encoding_dim):
        super(BCritic, self).__init__()
        self.QText = nn.Sequential(
            nn.Linear(text_input_dim + 2, text_input_dim),
            nn.ReLU(),
            nn.Linear(text_input_dim, encoding_dim),
            nn.Tanh()
        )
        self.QImage = nn.Sequential(
            nn.Linear(image_input_dim + 2, image_input_dim),
            nn.ReLU(),
            nn.Linear(image_input_dim, encoding_dim),
            nn.Tanh()
        )
        self.QScore = nn.Sequential(
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, 1)
        )

    def forward(self, text_s, text_a, image_s, image_a):
        text_rep = self.QText(torch.cat([text_s, text_a], dim=-1))
        image_rep = self.QImage(torch.cat([image_s, image_a], dim=-1))
        score = self.QScore(torch.cat([text_rep, image_rep], dim=-1))
        return score


class MultiAgentClassifier():
    def __init__(self, vocab_size, word2id):
        super(MultiAgentClassifier, self).__init__()
        self.textEncoder = TextEncoder(input_dim=128,
                                       encoding_dim=128,
                                       vocab_size=vocab_size,
                                       word2id=word2id)
        self.imageEncoder = ImageEncoder(input_dim=7 * 7 * 512,
                                         encoding_dim=128)
        self.textActor = TextActor(encoding_dim=128)
        self.imageActor = ImageActor(encoding_dim=128)
        self.targetTextActor = TextActor(encoding_dim=128)
        self.targetImageActor = ImageActor(encoding_dim=128)
        self.imageActor = ImageActor(encoding_dim=128)
        self.critic = Critic(text_input_dim=128 * 3,
                             image_input_dim=128 * 3,
                             encoding_dim=128)
        self.targetCritic = Critic(text_input_dim=128 * 3,
                                   image_input_dim=128 * 3,
                                   encoding_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(self.textActor.encoding_dim + self.imageActor.encoding_dim,
                      self.textActor.encoding_dim + self.imageActor.encoding_dim),
            nn.ReLU(),
            nn.Linear(self.textActor.encoding_dim + self.imageActor.encoding_dim,
                      2)
        )
        self.textEncoder.to(device)
        self.imageEncoder.to(device)
        self.textActor.to(device)
        self.targetTextActor.to(device)
        self.imageActor.to(device)
        self.targetImageActor.to(device)
        self.critic.to(device)
        self.targetCritic.to(device)
        self.classifier.to(device)
        self.textEncoderOptimizer = optim.Adam(self.textEncoder.parameters(),
                                               lr=0.001, weight_decay=1e-6)
        self.imageEncoderOptimizer = optim.Adam(self.imageEncoder.parameters(),
                                                lr=0.001, weight_decay=1e-6)
        self.textActorOptimizer = optim.Adam(self.textActor.parameters(),
                                             lr=0.001, weight_decay=1e-6)
        self.imageActorOptimizer = optim.Adam(self.imageActor.parameters(),
                                              lr=0.001, weight_decay=1e-6)
        self.criticOptimizer = optim.Adam(self.critic.parameters(),
                                          lr=0.001, weight_decay=1e-6)
        self.classifierOptimizer = optim.Adam(self.classifier.parameters(),
                                              lr=0.001, weight_decay=1e-6)
        self.classifierLossFunction = nn.CrossEntropyLoss()
        self.buffer = Buffer(max_len=100000)
        # bind target buffer to source buffer, for list is a reference type
        self.targetTextActor.selected_data = self.textActor.selected_data
        self.targetTextActor.unselected_data = self.textActor.unselected_data
        self.targetImageActor.selected_data = self.imageActor.selected_data
        self.targetImageActor.unselected_data = self.imageActor.unselected_data
        self.hard_update()

    def update_buffer(self, user, label, need_backward=True, train_classifier=True, update_buffer=True):
        # max_r = -1
        last_score = 0.5
        tweet_num = len(user.tweets)
        ground_truth = torch.Tensor(label).long().to(device)
        label = label[0]
        output = None
        selected_pair = []
        unselected_pair = []
        for i in range(tweet_num-1):

            #modified 
            tweet = user.tweets[i][0]
            image_feature = user.tweets[i][1]

            _tweet = tweet
            origin_image_feature = image_feature.data

            #modified
            next_tweet = user.tweets[i+1][0]
            next_image_feature = user.tweets[i+1][1]

            tweet = self.textEncoder([tweet])
            next_tweet = self.textEncoder([next_tweet]).detach()
            image_feature = self.imageEncoder(image_feature.to(device))
            next_image_feature = self.imageEncoder(next_image_feature.to(device)).detach()
            text_s1, text_a1 = self.textActor(tweet,
                                              need_backward=need_backward,
                                              target=False)
            image_s1, image_a1 = self.imageActor(image_feature,
                                                 need_backward=need_backward,
                                                 target=False)
            _t = _tweet if text_a1 == 1 else [' ']
            _i = origin_image_feature if image_a1 == 1 else torch.zeros(1, 7 * 7* 512)
            _o_t = _tweet if text_a1 == 0 else [' ']
            _o_i = origin_image_feature if image_a1 == 0 else torch.zeros(1, 7 * 7 * 512)
            selected_pair.append((_t, _i))
            unselected_pair.append((_o_t, _o_i))
            selectedText = torch.cat(self.textActor.selected_data, dim=0)
            selectedText = torch.mean(selectedText, dim=0, keepdim=True)
            selectedImage = torch.cat(self.imageActor.selected_data, dim=0)
            selectedImage = torch.mean(selectedImage, dim=0, keepdim=True)
            output = self.classifier(torch.cat([selectedText, selectedImage], dim=1))
            score = F.softmax(output, dim=-1)[0, label].item()
            r = score - last_score
            # if max_r < r and torch.sum(origin_image_feature).item() != 0:
            #     max_image_feat = origin_image_feature
            #     max_r = r
            #     with open(f'indicator_pic/{tweet_id}.pkl', 'wb') as f:
            #         pickle.dump(max_image_feat, f)
            last_score = score
            text_s2, _ = self.targetTextActor(next_tweet,
                                                 need_backward=False,
                                                 target=True)
            image_s2, _ = self.targetImageActor(next_image_feature,
                                                       need_backward=False,
                                                       target=True)
            text_a1 = [[1., 0.]] if text_a1 == 0 else [[0., 1.]]
            opposed_text_a = [[text_a1[0][1], text_a1[0][0]]]
            image_a1 = [[1., 0.]] if image_a1 == 0 else [[0., 1.]]
            opposed_image_a = [[image_a1[0][1], image_a1[0][0]]]
            text_a1 = torch.Tensor(text_a1).to(device)
            opposed_text_a = torch.Tensor(opposed_text_a).to(device)
            image_a1 = torch.Tensor(image_a1).to(device)
            opposed_image_a = torch.Tensor(opposed_image_a).to(device)
            if update_buffer:
                self.buffer.append(text_s1, image_s1, text_a1, image_a1, r, text_s2, image_s2, opposed_text_a, opposed_image_a)


        classification_loss = self.classifierLossFunction(output, ground_truth)
        prediction = torch.max(output, 1, keepdim=False)[1].item()
        prob = F.softmax(output, dim=-1)[0, label].item()

        if train_classifier:
            self.textEncoderOptimizer.zero_grad()
            self.imageEncoderOptimizer.zero_grad()
            self.classifierOptimizer.zero_grad()
            classification_loss.backward()
            self.textEncoderOptimizer.step()
            self.imageEncoderOptimizer.step()
            self.classifierOptimizer.step()
        self.reset_actor_selected_and_unselected_data()
        # selected_pair.insert(0, 0)
        # unselected_pair.insert(0, 0)
        # root = 'positive' if label == 1 else 'negative'
        # with open(f'new_ds/selected_pair/{root}/{index}.pkl', 'wb') as f:
        #     pickle.dump(selected_pair, f)
        # with open(f'new_ds/unselected_pair/{root}/{index}.pkl', 'wb') as f:
        #     pickle.dump(unselected_pair, f)
        return classification_loss.item(), prediction, prob

    def optimize_AC(self):
        # update critic
        batch_text_s, batch_image_s, \
        batch_text_a, batch_image_a, \
        batch_r, \
        batch_text_next_s, batch_image_next_s, \
        batch_opposed_text_a, \
        batch_opposed_image_a = self.buffer.sample(1024)
        Q_predicted = self.critic(batch_text_s, batch_text_a,
                         batch_image_s, batch_image_a)
        batch_text_next_a = self.targetTextActor.choose_action(batch_text_next_s,
                                                               need_backward=False,
                                                               target=True)
        batch_image_next_a = self.targetImageActor.choose_action(batch_image_next_s,
                                                                 need_backward=False,
                                                                 target=True)
        Q2 = self.targetCritic(batch_text_next_s, batch_text_next_a,
                               batch_image_next_s, batch_image_next_a)
        Q_expected = batch_r + 0.99 * Q2.detach()
        loss_critic = F.smooth_l1_loss(Q_predicted, Q_expected)
        self.criticOptimizer.zero_grad()
        loss_critic.backward()
        self.criticOptimizer.step()
        self.soft_update(0.1, type='critic')
        # update actor
        text_A  = self.targetCritic(batch_text_s, batch_text_a,
                              batch_image_s, batch_image_a) \
                  - \
                  self.targetCritic(batch_text_s, batch_opposed_text_a,
                              batch_image_s, batch_image_a)
        image_A = self.targetCritic(batch_text_s, batch_text_a,
                              batch_image_s, batch_image_a) \
                  - \
                  self.targetCritic(batch_text_s, batch_text_a,
                              batch_image_s, batch_opposed_image_a)
        # Adventage = self.critic(batch_text_s, batch_text_a, batch_image_s, batch_image_a).data - batch_r + 0.99 * self.targetCritic(batch_text_s, batch_text_a, batch_image_s, batch_image_a).data
        text_log_probs = self.textActor.get_log_probs(batch_text_s, batch_text_a[:, 1])
        image_log_probs = self.imageActor.get_log_probs(batch_image_s, batch_image_a[:, 1])
        loss_actor = -torch.sum(text_log_probs * text_A.detach() + image_log_probs * image_A.detach())
        self.textActorOptimizer.zero_grad()
        self.imageActorOptimizer.zero_grad()
        loss_actor.backward()
        self.textActorOptimizer.step()
        self.imageActorOptimizer.step()
        self.soft_update(0.1, type='actor')

    def reset_actor_selected_and_unselected_data(self):
        self.textActor.selected_data = [torch.zeros(1, self.textActor.encoding_dim).to(device)]
        self.textActor.unselected_data = [torch.zeros(1, self.textActor.encoding_dim).to(device)]
        self.imageActor.selected_data = [torch.zeros(1, self.imageActor.encoding_dim).to(device)]
        self.imageActor.unselected_data = [torch.zeros(1, self.imageActor.encoding_dim).to(device)]

        self.targetTextActor.selected_data = self.textActor.selected_data
        self.targetTextActor.unselected_data = self.textActor.unselected_data
        self.targetImageActor.selected_data = self.imageActor.selected_data
        self.targetImageActor.unselected_data = self.imageActor.unselected_data

    def hard_update(self):
        for target_param, param in zip(self.targetCritic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.targetTextActor.parameters(),
                                       self.textActor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.targetImageActor.parameters(),
                                       self.imageActor.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, scale, type):
        if type == 'critic':
            for target_param, param in zip(self.targetCritic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(param.data * scale + target_param.data * (1 - scale))
        if type == 'actor':
            for target_param, param in zip(self.targetTextActor.parameters(),
                                           self.textActor.parameters()):
                target_param.data.copy_(target_param.data)
            for target_param, param in zip(self.targetImageActor.parameters(),
                                           self.imageActor.parameters()):
                target_param.data.copy_(target_param.data)
        if type == 'all':
            for target_param, param in zip(self.targetCritic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(param.data * scale + target_param.data * (1 - scale))
            for target_param, param in zip(self.targetTextActor.parameters(),
                                           self.textActor.parameters()):
                target_param.data.copy_(target_param.data)
            for target_param, param in zip(self.targetImageActor.parameters(),
                                           self.imageActor.parameters()):
                target_param.data.copy_(target_param.data)

    def pre_train_classifier(self, user, label, need_backward=True):

        #user.tweets is a list of tuple (tweets, image_feature)
        tweets = [sample[0] for sample in user.tweets]

        
        image_features = torch.cat([sample[1].to(device) for sample in user.tweets], dim=0)


        text_encodings = self.textEncoder(tweets)
        image_encodings = self.imageEncoder(image_features)
        text_output = torch.mean(text_encodings, dim=0, keepdim=True)
        image_output = torch.mean(image_encodings, dim=0, keepdim=True)
        output = self.classifier(torch.cat([text_output, image_output], dim=-1))
        prediction = torch.max(output, 1, keepdim=False)[1].item()
        label = torch.Tensor(label).long().to(device)
        classifier_loss = self.classifierLossFunction(output, label)
        if need_backward:
            self.textEncoderOptimizer.zero_grad()
            self.imageEncoderOptimizer.zero_grad()
            self.classifierOptimizer.zero_grad()
            classifier_loss.backward()
            self.textEncoderOptimizer.step()
            self.imageEncoderOptimizer.step()
            self.classifierOptimizer.step()
        return classifier_loss.item(), prediction

    def joint_train(self, user, label):
        cf_loss, prediction, prob = self.update_buffer(user, label, need_backward=True, train_classifier=True, update_buffer=True)
        self.optimize_AC()
        return  cf_loss, prediction


class DAN(nn.Module):
    def __init__(self, vocab_size, word2id, embedding_size, fix_length):
        super(DAN, self).__init__()
        self.word2id = word2id
        self.embedding_size = embedding_size
        self.fix_length = fix_length
        self.embeddingLayer = nn.Embedding(vocab_size+2, embedding_dim=embedding_size, padding_idx=vocab_size+1)
        self.textEncoder = nn.LSTM(input_size=embedding_size,
                                   hidden_size=embedding_size//2,
                                   bidirectional=True,
                                   num_layers=1,
                                   batch_first=True)
        self.reshapeLayer = lambda tensor: tensor.view(49, 512)
        # parameters for Visual Attention.
        self.W_v = nn.Linear(512, embedding_size, bias=False)
        self.W_vm = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_vh = nn.Linear(embedding_size, 49, bias=False)
        self.P = nn.Linear(512, embedding_size, bias=False)
        # parameters for Textual Attention.
        self.W_u = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_um = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_uh = nn.Linear(embedding_size, fix_length, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 2)
        )

    def forward(self, user):
        # tweet_num = len(user.tweets)
        if len(user[0].tweets) > 100 :
            user[0].tweets = user[0].tweets[:100]
        m = [self.dual_att(tweet, image.to(device)) for tweet, image in user[0].tweets]
        m = torch.mean(torch.cat(m, dim=0), dim=0, keepdim=True)
        output = self.classifier(m)
        return output

    def dual_att(self, tweet, image):
        tweetEmbedding = self.embedding_lookup(tweet)
        u, (_, _) = self.textEncoder(tweetEmbedding)
        u = u.squeeze(0)
        u0 = torch.mean(u, dim=0, keepdim=True)
        v = self.reshapeLayer(image)
        v0 = F.tanh(self.P(torch.mean(v, dim=0, keepdim=True)))
        m0 = u0 * v0
        m_u = m0.detach()
        m_v = m0.detach()
        for u_t in u:
            h_ut = F.tanh(self.W_u(u_t)) * F.tanh(self.W_um(m_u))
            a_ut = F.softmax(self.W_uh(h_ut), dim=-1)
            u_k = torch.matmul(a_ut, u)
            m_u = m_u + u_k
        for v_n in v:
            h_vn = F.tanh(self.W_v(v_n)) * F.tanh(self.W_vm(m_v))
            a_vn = F.softmax(self.W_vh(h_vn), dim=-1)
            v_k = F.tanh(self.P(torch.matmul(a_vn, v)))
            m_v = m_v + v_k
        return torch.cat([m_u, m_v], dim=-1)

    def embedding_lookup(self, tweet):
        id = []
        for i, word in enumerate(tweet):
            if i == self.fix_length:
                break
            if word in self.word2id:
                id.append(self.word2id[word])
            else:
                id.append(self.embeddingLayer.padding_idx - 1)
        id += [self.embeddingLayer.padding_idx] * (self.fix_length - len(id))
        id = torch.Tensor([id]).long().to(device)
        embeds = self.embeddingLayer(id)
        return embeds


class CAN(nn.Module):
    def __init__(self, vocab_size, word2id, embedding_size, fix_length):
        super(CAN, self).__init__()
        self.word2id = word2id
        self.embedding_size = embedding_size
        self.fix_length = fix_length
        self.embeddingLayer = nn.Embedding(vocab_size + 2, embedding_dim=embedding_size, padding_idx=vocab_size + 1)
        self.textEncoder = nn.LSTM(input_size=embedding_size,
                                   hidden_size=embedding_size,
                                   bidirectional=False,
                                   num_layers=1,
                                   batch_first=True)
        self.reshapeLayer = lambda tensor: tensor.view(49, 512)
        self.imageConvertLayer = nn.Linear(512, 100)
        self.W_VI = nn.Linear(100, 50, bias=False)
        self.W_VT = nn.Linear(100, 50, bias=False)
        self.W_PI = nn.Linear(100, 1, bias=True)

        self.W_V_I = nn.Linear(100, 50, bias=False)
        self.W_T = nn.Linear(100, 50, bias=False)
        self.W_PT = nn.Linear(100, 1, bias=True)

        self.W_f = nn.Linear(100, 2)

    def forward(self, user):
        user = user[0]
        probs = []
        for tweet, image in user.tweets:
            tweetEmbeds = self.embedding_lookup(tweet)
            VT, (_, _) = self.textEncoder(tweetEmbeds)
            VT = VT.squeeze(0)
            VI = self.imageConvertLayer(self.reshapeLayer(image.to(device)))
            v_I = self.tweet_guided_att(VI, VT)
            v_T = self.image_guided_att(VT, v_I)
            f = v_I + v_T
            p = self.W_f(f)
            probs.append(p)
        probs = torch.cat(probs, dim=0)
        output = torch.mean(probs, dim=0, keepdim=True)
        return output

    def tweet_guided_att(self, VI, VT):
        a = torch.mean(VT, dim=0, keepdim=True)
        vT = self.W_VT(a)
        vT = vT.repeat(VI.size(0), 1)
        vI = self.W_VI(VI)
        hI = F.tanh(torch.cat([vI, vT], dim=1))
        pI = F.softmax(self.W_PI(hI), dim=-1)
        v_I = torch.matmul(pI.permute(1, 0), VI)
        return v_I

    def image_guided_att(self, VT, v_I):
        vI = self.W_V_I(v_I)
        vI = vI.repeat(VT.size(0), 1)
        vT = self.W_T(VT)
        hT = F.tanh(torch.cat([vI, vT], dim=1))
        pT = F.softmax(self.W_PT(hT), dim=-1)
        v_T = torch.matmul(pT.permute(1, 0), VT)
        return v_T


    def embedding_lookup(self, tweet):
        id = []
        for i, word in enumerate(tweet):
            if i == self.fix_length:
                break
            if word in self.word2id:
                id.append(self.word2id[word])
            else:
                id.append(self.embeddingLayer.padding_idx - 1)
        id += [self.embeddingLayer.padding_idx] * (self.fix_length - len(id))
        id = torch.Tensor([id]).long().to(device)
        embeds = self.embeddingLayer(id)
        return embeds


class MAN(nn.Module):
    def __init__(self, vocab_size, word2id, embedding_size):
        super(MAN, self).__init__()
        self.word2id = word2id
        self.embedding_size = embedding_size
        self.embeddingLayer = nn.Embedding(vocab_size + 2, embedding_dim=embedding_size, padding_idx=vocab_size + 1)
        self.textEncoder = nn.LSTM(input_size=embedding_size,
                                   hidden_size=embedding_size//2,
                                   bidirectional=True,
                                   num_layers=1,
                                   batch_first=True)
        self.reshapeLayer = lambda tensor: tensor.view(49, 512)
        self.imageConvertLayer = nn.Linear(512, embedding_size)
        self.W_I = nn.Linear(embedding_size, 1, bias=False)
        self.W_T = nn.Linear(embedding_size, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 2)
        )

    def forward(self, user):
        user = user[0]
        features = []
        for tweet, image in user.tweets:
            tweetEmbeds = self.embedding_lookup(tweet)
            _, (text_h, _) = self.textEncoder(tweetEmbeds)
            image_h = self.imageConvertLayer(torch.mean(self.reshapeLayer(image.to(device)), dim=0, keepdim=True))
            text_h = text_h.view(1, -1)
            text_s = self.W_T(text_h.view(1, -1))
            image_s = self.W_I(image_h)
            att = F.softmax(torch.cat([text_s, image_s], dim=1), dim=1)
            feature = torch.matmul(att, torch.cat([text_h, image_h], dim=0))
            features.append(feature)
        mean_feature = torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)
        output = self.classifier(mean_feature)
        return output


    def embedding_lookup(self, tweet):
        id = []
        for i, word in enumerate(tweet):
            if word in self.word2id:
                id.append(self.word2id[word])
            else:
                id.append(self.embeddingLayer.padding_idx - 1)
        id = torch.Tensor([id]).long().to(device)
        embeds = self.embeddingLayer(id)
        return embeds