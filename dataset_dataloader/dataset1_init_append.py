import os


class MyDataset:
    def __init__(self, text, label, batch_size):
        self.text = text
        self.label = label
        self.batch_size = batch_size

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
        return self

    def __next__(self):
        if self.cursor >= len(self):
            raise StopIteration
        batch_text = []
        batch_label = []
        for i in range(self.batch_size):
            if self.cursor + batch_size > len(self):
                '''returns the remaining elements'''
                batch_text, batch_label = self[self.cursor:]
                self.cursor = len(self)
                break
            else:
                '''returns the current element'''
                text, label = self[self.cursor]
                batch_text.append(text)
                batch_label.append(label)
                self.cursor += 1
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
    batch_size = 3
    epoch = 10
    data = MyDataset(text, label, batch_size)
    for e in range(epoch):
        for batch_text, batch_label in data:
            print(batch_text, batch_label)
