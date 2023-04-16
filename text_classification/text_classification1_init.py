import os
from collections import defaultdict
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
        text = self.all_text[index]
        label = self.all_label[index]
        text_len = len(text)
        return text, label, text_len

    def process_batch_batch(self, data):
        batch_text = []
        batch_label = []
        batch_len = []
        for d in data:
            batch_text.append(d[0])
            batch_label.append(d[1])
            batch_len.append(d[2])
        min_len = min(batch_len)
        batch_text = [i[:min_len] for i in batch_text]
        return batch_text, batch_label

    def __len__(self):
        return len(self.all_text)


if __name__ == "__main__":
    file_path = os.path.join("..", "data", "text_classification", "data.txt")
    train_text, train_label = get_data(file_path)

    word_2_index = build_word_2_index(train_text)

    train_batch_size = 4
    epoch = 1

    train_dataset = TextDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False,
                                  collate_fn=train_dataset.process_batch_batch)
    for e in range(epoch):
        for batch_text, batch_label in train_dataloader:
            print("*" * 100)
            print(f"text:{batch_text}\nlabel:{batch_label}")
