import os

import numpy as np


class MyDataset:
    def __init__(self, text, label, batch_size, shuffle):
        self.text = np.array(text)
        self.label = np.array(label)
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert len(text) == len(label), \
            "Inconsistent number of samples and labels!"

    def __iter__(self):
        return MyDataLoader(self)

    def __len__(self):
        return len(self.text)


class MyDataLoader:
    def __init__(self, dataset):
        self.cursor = 0
        self.dataset = dataset
        self.shuffle_index = np.arange(len(dataset))
        if self.dataset.shuffle:
            np.random.shuffle(self.shuffle_index)

    def __getitem__(self, index):
        global max_len, word_2_index
        text = self.dataset.text[index][:max_len]  # 1.裁剪
        label = self.dataset.label[index]
        text_idx = [word_2_index[i] for i in text]  # 2.word to index;
        text_idx = text_idx + [0] * (max_len - len(text))  # 3.填充
        return text, text_idx, label

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        batch_text = []
        batch_label = []
        batch_text_idx = []
        batch_index = self.shuffle_index[self.cursor:self.cursor + self.dataset.batch_size]
        for index in batch_index:
            text, text_idx, label = self[index]
            batch_text.append(text)
            batch_text_idx.append(text_idx)
            batch_label.append(label)
        self.cursor += self.dataset.batch_size
        return batch_text, np.array(batch_text_idx), np.array(batch_label)


class MyModel:
    def __init__(self):
        self.model = np.random.uniform(low=-1.0, high=1.0, size=(max_len, 1))

    def forward(self, batch_idx):
        pre = batch_idx @ self.model
        return pre


def get_data(file_path):
    dict_data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = line.strip()
            text = data.split(' ')[0]
            label = data.split(' ')[1]
            dict_data[text] = label
        f.close()
    text = list(dict_data.keys())
    label = list(dict_data.values())
    return text, label


def build_word_2_index(train_text):
    word_2_index = {"<PAD>": 0}
    for text in train_text:
        for w in text:
            if w not in word_2_index:
                word_2_index[w] = len(word_2_index)
    return word_2_index


if __name__ == "__main__":
    file_path = os.path.join("..", "data", "word_2_index", "train.txt")
    text, label = get_data(file_path)

    word_2_index = build_word_2_index(text)

    shuffle = True
    epoch = 10
    batch_size = 2
    max_len = 10

    data = MyDataset(text, label, batch_size, shuffle)
    loader = MyDataLoader(data)
    model = MyModel()

    i = 0
    for e in range(epoch):
        print("\n", "*" * 25, f"epoch{i}", "*" * 25, "\n")
        i += 1
        j = 0
        for batch_text, batch_text_idx, batch_label in data:
            print("=" * 25, f"batch{j}", "=" * 25)
            print("batch_text[{}]:{};\nbatch_text_idx:{};\nbatch_label:{}".format(j,
                                                                                  batch_text,
                                                                                  batch_text_idx,
                                                                                  batch_label))
            predict = model.forward(batch_text_idx)
            print(predict)
            j += 1
