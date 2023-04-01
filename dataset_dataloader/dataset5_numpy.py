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

        self.cursor = 0

    def __iter__(self):
        self.cursor = 0
        self.shuffle_index = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(self.shuffle_index)
        return self

    def __next__(self):
        if self.cursor >= len(self):
            raise StopIteration
        batch_index = self.shuffle_index[self.cursor:self.cursor + self.batch_size]
        batch_text = self.text[batch_index]
        batch_label = self.label[batch_index]
        self.cursor += batch_size
        return batch_text, batch_label

    def __len__(self):
        return len(self.text)


def get_data():
    dict_data = {}
    file_path = os.path.join("..", "data", "dataset_dataloader", "train.txt")
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
    text, label = get_data()
    print(text, label)
    batch_size = 2
    epoch = 10
    shuffle = True
    data = MyDataset(text, label, batch_size, shuffle)
    for e in range(epoch):
        print("\n", "*" * 50, "\n")
        for batch_text, batch_label in data:
            print(batch_text, batch_label)
