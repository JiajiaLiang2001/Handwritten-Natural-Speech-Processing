import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_data(path):
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
            text, label = data.split(" ")
            label = int(label)
            all_text.append(text)
            all_label.append(label)
        except (ValueError, IndexError):
            pass
    return all_text, all_label


def build_word_2_index(train_text):
    '''
    根据训练文本构建单词到索引的映射关系字典
    :param train_text:训练文本数据，包含多个文本字符串
    :return:词到索引的映射关系字典，使用 defaultdict 实现，其中默认值为当前字典长度
    '''
    word_2_index = defaultdict(lambda: len(word_2_index))
    word_2_index["<PAD>"] = 0
    for text in train_text:
        words = set(text)
        for w in words:
            _ = word_2_index[w]
    return word_2_index


class TextDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        global word_2_index
        text = self.all_text[index]
        text_len = len(text)
        text_index = [word_2_index[i] for i in text]
        label = self.all_label[index]
        return text_index, label, text_len

    def process_batch_batch(self, data):
        global max_len
        batch_text = []
        batch_label = []
        batch_len = []
        for d in data:
            batch_text.append(d[0])
            batch_label.append(d[1])
            batch_len.append(d[2])
        batch_text = [i[:max_len] for i in batch_text]
        batch_text = [i + [0] * (max_len - len(i)) for i in batch_text]
        return torch.tensor(batch_text, dtype=torch.float32), torch.tensor(batch_label)

    def __len__(self):
        return len(self.all_text)


class Model(nn.Module):
    def __init__(self, feature_num, class_num):
        super().__init__()
        self.linear = nn.Linear(feature_num, class_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        y = self.linear.forward(x)
        if label is not None:
            loss = self.loss(y, label)
            return loss
        else:
            return torch.argmax(y, dim=1)


if __name__ == "__main__":
    file_path = os.path.join("..", "data", "text_classification", "data.txt")
    train_text, train_label = get_data(file_path)

    word_2_index = build_word_2_index(train_text)

    train_batch_size = 2
    epoch = 10
    lr = 0.01
    max_len = 10

    train_dataset = TextDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False,
                                  collate_fn=train_dataset.process_batch_batch)

    model = Model(feature_num=max_len, class_num=len(set(train_label)))
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for e in range(epoch):
        print(f"{'*' * 50}_epoch:{e}_{'*' * 50}")
        for batch_text, batch_label in train_dataloader:
            loss = model.forward(batch_text, batch_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"loss:{loss:.2f}")
