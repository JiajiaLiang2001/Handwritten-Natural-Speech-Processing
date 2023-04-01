import numpy as np


class MyDataset:
    def __init__(self, x, y, batch_size, shuffle):
        self.x = np.array(x)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return MyDataLoader(self)

    def __len__(self):
        return len(self.x)


class MyDataLoader:
    def __init__(self, dataset):
        self.cursor = 0
        self.dataset = dataset
        self.index = np.arange(len(dataset))
        if self.dataset.shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        batch_index = self.index[self.cursor:self.cursor + self.dataset.batch_size]
        batch_x = self.dataset.x[batch_index]
        batch_y = self.dataset.y[batch_index]
        self.cursor += self.dataset.batch_size
        return batch_x, batch_y
