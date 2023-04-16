import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_data(path, num):
    '''
    从文件中读取数据，并返回文本和标签的列表
    :param path:文件路径
    :return:
        - all_text (list): 文本列表
        - all_label (list): 标签列表
    '''
    all_text = []
    all_label = []
    with open(path, "r", encoding="utf8") as f:
        all_data = f.read().split("\n")
    for data in all_data:
        try:
            text, label = data.split("\t")
            label = int(label)
            all_text.append(text)
            all_label.append(label)
        except (ValueError, IndexError):
            pass
    if num is None:
        return all_text, all_label
    else:
        return all_text[:num], all_label[:num]
    return all_text, all_label


def build_word_2_index(train_text):
    '''
    根据训练文本构建单词到索引的映射关系字典
    :param train_text:训练文本数据，包含多个文本字符串
    :return:词到索引的映射关系字典，使用 defaultdict 实现，其中默认值为当前字典长度
    '''
    word_2_index = defaultdict(lambda: len(word_2_index))
    word_2_index["<PAD>"] = 0
    word_2_index["<UNK>"] = 1
    for text in train_text:
        words = set(text)
        for w in words:
            _ = word_2_index[w]
    return word_2_index


def calculation_accuracy(dataloader, model):
    right_num = 0
    total_num = 0
    for batch_text, batch_label in dataloader:
        batch_text = batch_text.to(device)
        batch_label = batch_label.to(device)
        predict_label = model(batch_text)
        right_num += int(torch.sum(predict_label == batch_label))
        total_num += batch_label.size(0)
    accuracy = (right_num / total_num) * 100
    return accuracy


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TextDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        global word_2_index
        text = self.all_text[index]
        text_len = len(text)
        text_index = [word_2_index.get(i, 1) for i in text]
        label = self.all_label[index]
        return text_index, label, text_len

    def process_batch_batch(self, data):
        global max_len, word_2_index, index_2_embedding
        batch_text = [d[0][:max_len] + [0] * (max_len - len(d[0])) for d in data]
        batch_text = torch.tensor([i + [0] * (max_len - len(i)) for i in batch_text])
        batch_label = torch.tensor([d[1] for d in data], dtype=torch.int64)
        return batch_text, batch_label

    def __len__(self):
        return len(self.all_text)


class Model(nn.Module):
    def __init__(self, words_size, embeding_dim, class_num):
        super().__init__()
        self.embedding = torch.nn.Embedding(words_size, embeding_dim)
        self.linear = nn.Linear(embeding_dim, class_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = self.embedding(x)
        y = self.linear.forward(x)
        y = torch.mean(y, dim=1)
        if label is not None:
            loss = self.loss(y, label)
            return loss
        else:
            return torch.argmax(y, dim=-1)


if __name__ == "__main__":
    same_seeds(1000)
    train_file_path = os.path.join("..", "data", "text_classification", "train.txt")
    train_text, train_label = get_data(train_file_path, 5000)
    dev_file_path = os.path.join("..", "data", "text_classification", "dev.txt")
    dev_text, dev_lable = get_data(dev_file_path, 2000)

    word_2_index = build_word_2_index(train_text)

    train_batch_size = 100
    epoch = 10
    lr = 0.001
    max_len = 30
    words_num = len(word_2_index)
    embeding_num = 700
    labels_num = len(set(train_label))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = TextDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  collate_fn=train_dataset.process_batch_batch)
    dev_dataset = TextDataset(dev_text, dev_lable)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=10,
                                shuffle=False,
                                collate_fn=dev_dataset.process_batch_batch)

    model = Model(words_size=words_num, embeding_dim=embeding_num, class_num=labels_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for e in range(epoch):
        print(f"{'*' * 50}_epoch:{e}_{'*' * 50}")
        for batch_text, batch_label in train_dataloader:
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            loss = model.forward(batch_text, batch_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"loss:{loss:.2f}")

        print(f"accuracy:{calculation_accuracy(dev_dataloader, model):.2f}%")
