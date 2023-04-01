import os
import random


class MyDataset:
    def __init__(self, text, label, batch_size, shuffle):
        self.text = text
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert len(text) == len(label), \
            "Inconsistent number of samples and labels!"

        self.cursor = 0

    def __getitem__(self, index):
        text = label = None
        try:
            text = self.text[index]
            label = self.label[index]
        except IndexError:
            print("Data index out of range")
        finally:
            return text, label

    def __iter__(self):
        self.cursor = 0
        self.shuffle_index = [i for i in range(len(self))]
        if self.shuffle:
            random.shuffle(self.shuffle_index)
        return self

    def __next__(self):
        if self.cursor >= len(self):
            raise StopIteration
        batch_text = []
        batch_label = []
        for i in range(self.batch_size):
            index = self.shuffle_index[self.cursor]
            if self.cursor + batch_size > len(self):
                '''returns the remaining elements'''
                for i in range(len(self) - self.cursor):
                    batch_text, batch_label = self.append_data(index, batch_text, batch_label)
                    if self.cursor == len(self):
                        break
                    index = self.shuffle_index[self.cursor]
                break
            else:
                '''returns the current element'''
                batch_text, batch_label = self.append_data(index, batch_text, batch_label)
        return batch_text, batch_label

    def __len__(self):
        return len(self.text)

    def append_data(self, index, batch_text, batch_label):
        text, label = self[index]
        batch_text.append(text)
        batch_label.append(label)
        self.cursor += 1
        return batch_text, batch_label


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
    batch_size = 3
    epoch = 10
    shuffle = True
    data = MyDataset(text, label, batch_size, shuffle)
    for e in range(epoch):
        print("\n", "*" * 50, "\n")
        for batch_text, batch_label in data:
            print(batch_text, batch_label)
