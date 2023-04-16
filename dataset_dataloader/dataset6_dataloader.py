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

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        batch_index = self.shuffle_index[self.cursor:self.cursor + self.dataset.batch_size]
        batch_text = self.dataset.text[batch_index]
        batch_label = self.dataset.label[batch_index]
        self.cursor += self.dataset.batch_size
        return batch_text, batch_label


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


if __name__ == "__main__":
    file_path = os.path.join("..", "data", "dataset_dataloader", "train.txt")
    text, label = get_data(file_path)
    print(text, label)
    batch_size = 2
    epoch = 10
    shuffle = True
    data = MyDataset(text, label, batch_size, shuffle)
    loader = MyDataLoader(data)
    for e in range(epoch):
        print("\n", "*" * 50, "\n")
        for batch_text, batch_label in data:
            print(batch_text, batch_label)
