import os
import struct

import numpy as np
from dataset_dataloader.dataset_dataloader import MyDataset, MyDataLoader


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
    size = labels.shape[0]
    result = np.zeros((size, class_num))
    for index, label in enumerate(labels):
        result[index][label] = 1
    return result


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    return ex / sum_ex


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def calculation_accuracy(data, label, w1, w2, b1, b2, activation_function):
    h = data @ w1 + b1
    t = activation_function(h)
    y = t @ w2 + b2
    sy = softmax(y)
    p = np.argmax(sy, axis=1)
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
batch_size = 100
hidden_size = 256

w1 = np.random.normal(0, 1, size=(784, hidden_size))
w2 = np.random.normal(0, 1, size=(hidden_size, 10))
b1 = np.zeros((1, hidden_size))
b2 = np.zeros((1, 10))

dataset = MyDataset(train_data, train_label, batch_size=batch_size, shuffle=True)

for e in range(epoch):
    print(f"{'*' * 50}_epoch:{e}_{'*' * 50}")
    for X, y in dataset:
        H = X @ w1 + b1
        SH = sigmoid(H)
        Y = SH @ w2 + b2
        SY = softmax(Y)
        loss = -np.sum(y * np.log(SY)) / X.shape[0]

        G2 = (SY - y) / X.shape[0]
        G1 = G2 @ w2.T * SH * (1 - SH)  # G = G2 @ w2.T

        delta_w2 = SH.T @ G2
        delta_w1 = X.T @ G1

        delta_b2 = np.sum(G2, axis=0, keepdims=True)
        delta_b1 = np.sum(G1, axis=0, keepdims=True)

        w1 = w1 - delta_w1 * lr
        w2 = w2 - delta_w2 * lr
        b1 = b1 - delta_b1 * lr
        b2 = b2 - delta_b2 * lr

    accuracy = calculation_accuracy(validation_data, validation_label, w1, w2, b1, b2, sigmoid)
    print(f"acc:{accuracy:.2f}%")
