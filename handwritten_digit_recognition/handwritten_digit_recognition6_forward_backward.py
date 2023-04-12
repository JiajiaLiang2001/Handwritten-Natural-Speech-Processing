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


def calculation_accuracy(data, label, linear1_layer, linear2_layer, activation_layer):
    X = linear1_layer.forward(data)
    X = activation_layer(X)
    X = linear2_layer.forward(X)
    p = np.argmax(softmax(X), axis=1)
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


class Sigmoid(Module):
    def forward(self, H):
        SH = sigmoid(H)
        self.SH = SH
        return SH

    def backward(self, G):
        return G * self.SH * (1 - self.SH)


class Tanh(Module):
    def forward(self, H):
        TH = tanh(H)
        self.TH = TH
        return TH

    def backward(self, G):
        return G * (1 - self.TH ** 2)


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

'''
Activation function:sigmoid
'''
linear1_layer = Linear(in_feature=784, out_feature=hidden_size)
sigmoid_layer = Sigmoid()
linear2_layer = Linear(in_feature=hidden_size, out_feature=10)

print("The activation function is sigmoid......\n")

for e in range(epoch):
    print(f"{'*' * 50}_epoch:{e}_{'*' * 50}")
    for X, y in dataset:
        X = linear1_layer.forward(X)
        X = sigmoid_layer.forward(X)
        X = linear2_layer.forward(X)

        loss = -np.sum(y * np.log(softmax(X))) / X.shape[0]
        G = (softmax(X) - y) / X.shape[0]

        G = linear2_layer.backward(G)
        G = sigmoid_layer.backward(G)
        G = linear1_layer.backward(G)

    accuracy = calculation_accuracy(validation_data, validation_label, linear1_layer, linear2_layer,
                                    sigmoid_layer)
    print(f"acc:{accuracy:.2f}%")

'''
Activation function:tanh
'''
linear1_layer = Linear(in_feature=784, out_feature=hidden_size)
tanh_layer = Sigmoid()
linear2_layer = Linear(in_feature=hidden_size, out_feature=10)

print("The activation function is tanh......\n")

for e in range(epoch):
    print(f"{'*' * 50}_epoch:{e}_{'*' * 50}")
    for X, y in dataset:
        X = linear1_layer.forward(X)
        X = tanh_layer.forward(X)
        X = linear2_layer.forward(X)

        loss = -np.sum(y * np.log(softmax(X))) / X.shape[0]
        G = (softmax(X) - y) / X.shape[0]

        G = linear2_layer.backward(G)
        G = tanh_layer.backward(G)
        G = linear1_layer.backward(G)

    accuracy = calculation_accuracy(validation_data, validation_label, linear1_layer, linear2_layer, tanh_layer)
    print(f"acc:{accuracy:.2f}%")
