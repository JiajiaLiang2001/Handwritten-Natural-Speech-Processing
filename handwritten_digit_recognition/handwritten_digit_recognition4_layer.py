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


def calculation_accuracy(data, label, linear1, linear2, activation_function):
    h = linear1.forward(data)  # h = data @ w1 + b1
    t = activation_function(h)
    y = linear2.forward(t)  # y = t @ w2 + b2
    sy = softmax(y)
    p = np.argmax(sy, axis=1)
    num = 0
    for pl, tl in zip(p, label):
        if pl == tl:
            num += 1
    return num / len(data) * 100


class Module:
    pass


class Linear(Module):
    def __init__(self, in_feature, out_feature):
        self.W = np.random.normal(0, 1, size=(in_feature, out_feature))
        self.b = np.zeros((1, out_feature))

    def forward(self, X):
        self.A = X
        return X @ self.W + self.b

    def backward(self, G):
        delta_W = self.A.T @ G
        delta_b = np.sum(G, axis=0, keepdims=True)
        self.W -= delta_W * lr
        self.b -= delta_b * lr
        return G @ self.W.T


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

dataset = MyDataset(train_data, train_label, batch_size=batch_size, shuffle=True)

linear1 = Linear(in_feature=784, out_feature=hidden_size)
linear2 = Linear(in_feature=hidden_size, out_feature=10)

for e in range(epoch):
    print(f"{'*' * 50}_epoch:{e}_{'*' * 50}")
    for X, y in dataset:
        H = linear1.forward(X)  # H = X @ w1 + b1
        SH = sigmoid(H)
        Y = linear2.forward(SH)  # Y = TH @ w2 + b2
        SY = softmax(Y)
        loss = -np.sum(y * np.log(SY)) / X.shape[0]

        G2 = (SY - y) / X.shape[0]
        G = linear2.backward(G2)

        G1 = G * SH * (1 - SH)
        linear1.backward(G1)

    accuracy = calculation_accuracy(validation_data, validation_label, linear1, linear2, sigmoid)
    print(f"acc:{accuracy:.2f}%")
