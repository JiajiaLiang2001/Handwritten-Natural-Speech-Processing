import os
import struct

import numpy as np

from dataset_dataloader.dataset_dataloader import MyDataset


def load_labels(file):
    """
    Load digital labels
    :param file:
    :return:
    """
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):
    """
    Load digital pictures
    :param file:
    :return:
    """
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)


def onehot(labels, class_num=10):
    result = np.zeros((labels.shape[0], class_num))
    for index, label in enumerate(labels):
        result[index][label] = 1
    return result


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    return ex / sum_ex


def calculation_accuracy(data, label, w, b):
    y = data @ w + b
    p = np.argmax(y, axis=1)
    num = 0
    for pl, tl in zip(p, label):
        if pl == tl:
            num += 1
    return num / len(data) * 100


train_data = load_images(
    os.path.join("..", "data", "handwritten_digit_recognition", "train-images.idx3-ubyte")) / 255
train_label = onehot(load_labels(
    os.path.join("..", "data", "handwritten_digit_recognition", "train-labels.idx1-ubyte")))

validation_data = load_images(
    os.path.join("..", "data", "handwritten_digit_recognition", "t10k-images.idx3-ubyte")) / 255
validation_label = load_labels(
    os.path.join("..", "data", "handwritten_digit_recognition", "t10k-labels.idx1-ubyte"))

lr = 0.01
epoch = 10
batch_size = 20

w = np.random.normal(0, 1, size=(784, 10))
b = np.zeros((1, 10))

dataset = MyDataset(train_data, train_label, batch_size=batch_size, shuffle=False)

for e in range(epoch):
    print(f"{'*' * 50}_epoch:{e}_{'*' * 50}")
    for X, y in dataset:
        Y = X @ w + b
        SY = softmax(Y)
        loss = -np.mean(y * np.log(SY))

        G = (SY - y) / X.shape[0]
        delta_w = X.T @ G
        delta_b = np.mean(G)

        w = w - delta_w * lr
        b = b - delta_b * lr
        print(loss)

    accuracy = calculation_accuracy(validation_data, validation_label, w, b)
    print(f"acc:{accuracy:.2f}%")
