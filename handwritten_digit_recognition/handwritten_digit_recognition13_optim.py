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


def calculation_accuracy(data, label, model):
    X = model(data)
    p = np.argmax(X, axis=1)
    num = 0
    for pl, tl in zip(p, label):
        if pl == tl:
            num += 1
    return num / len(data) * 100


class Module():
    def __init__(self):
        self.info = self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        pass

    def parameters(self):
        ps = [value for value in self.__dict__.values() if isinstance(value, Parameter)]
        for value in self.__dict__.values():
            if "__iter__" in dir(value) and not isinstance(value, str):
                for i in value:
                    i_attrs = i.__dict__
                    ps.extend([i_value for i_value in i_attrs.values() if isinstance(i_value, Parameter)])
        return ps


class Linear(Module):
    def __init__(self, in_feature, out_feature):
        self.W = Parameter(np.random.normal(0, 1, size=(in_feature, out_feature)))
        self.b = Parameter(np.zeros((1, out_feature)))

    def forward(self, X):
        self.A = X
        return X @ self.W.weight + self.b.weight

    def backward(self, G):
        self.W.grad += self.A.T @ G
        self.b.grad += np.sum(G, axis=0, keepdims=True)
        return G @ self.W.weight.T


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


class Softmax(Module):
    def forward(self, x):
        return softmax(x)

    def backward(self, G):
        return G

class Model(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X, y=None):
        for layer in self.layers:
            X = layer(X)
        if y is not None:
            self.G = (X - y) / X.shape[0]
        else:
            return X

    def backward(self):
        G = self.G
        for layer in self.layers[::-1]:
            G = layer.backward(G)

    def __repr__(self):
        infos = []
        for layer in self.layers:
            infos.append(layer.info)
        return "\n".join(infos)


class Parameter():
    def __init__(self, x):
        self.weight = x
        self.grad = np.zeros_like(x)


class Optim:
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = 0


class SGD(Optim):
    def step(self):
        for param in self.params:
            param.weight -= self.lr * param.grad


class MSGD(Optim):
    def __init__(self, params, lr=0.1):
        super().__init__(params, lr)
        self.u = 0.9
        for param in self.params:
            param.last_grad = 0

    def step(self):
        for param in self.params:
            param.weight = param.weight - self.lr * (self.u * param.last_grad + (1 - self.u) * param.grad)
            param.last_grad = param.grad


class Adam(Optim):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, e=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = e
        for param in self.params:
            param.m = 0
            param.v = 0
        self.t = 0

    def step(self):
        self.t += 1
        for param in self.params:
            g_t = param.grad
            param.m = self.beta1 * param.m + (1 - self.beta1) * g_t
            param.v = self.beta2 * param.v + (1 - self.beta2) * g_t * g_t
            mt_ = param.m / (1 - self.beta1 ** self.t)
            vt_ = param.v / (1 - self.beta2 ** self.t)
            param.weight = param.weight - self.lr * mt_ / (np.sqrt(vt_) + self.e)


np.random.seed(1000)

train_data = load_images(
    os.path.join("..", "data", "handwritten_digit_recognition", "train-images.idx3-ubyte")) / 255
train_label = onehot(load_labels(
    os.path.join("..", "data", "handwritten_digit_recognition", "train-labels.idx1-ubyte")))

validation_data = load_images(
    os.path.join("..", "data", "handwritten_digit_recognition", "t10k-images.idx3-ubyte")) / 255
validation_label = load_labels(
    os.path.join("..", "data", "handwritten_digit_recognition", "t10k-labels.idx1-ubyte"))

epoch = 10
batch_size = 100
hidden_size = 256

dataset = MyDataset(train_data, train_label, batch_size=batch_size, shuffle=True)

layers = [
    Linear(784, hidden_size),
    Sigmoid(),
    Linear(hidden_size, 10),
    Softmax()
]
model = Model(layers)
sgd_opt = SGD(model.parameters(), 0.1)
optimizer = sgd_opt
for e in range(epoch):
    for X, y in dataset:
        model(X, y)
        model.backward()
        optimizer.step()
        optimizer.zero_grad()
    accuracy = calculation_accuracy(validation_data, validation_label, model)
print(f"Accuracy of Gradient Descent:{accuracy:.2f}%")

layers = [
    Linear(784, hidden_size),
    Sigmoid(),
    Linear(hidden_size, 10),
    Softmax()
]
model = Model(layers)
msgd_opt = MSGD(model.parameters(), 0.1)
optimizer = msgd_opt
for e in range(epoch):
    for X, y in dataset:
        model(X, y)
        model.backward()
        optimizer.step()
        optimizer.zero_grad()
    accuracy = calculation_accuracy(validation_data, validation_label, model)
print(f"Accuracy of Dynamic Gradient Descent:{accuracy:.2f}%")

layers = [
    Linear(784, hidden_size),
    Sigmoid(),
    Linear(hidden_size, 10),
    Softmax()
]
model = Model(layers)
adam_opt = Adam(model.parameters(), 0.001)
optimizer = adam_opt
for e in range(epoch):
    for X, y in dataset:
        model(X, y)
        model.backward()
        optimizer.step()
        optimizer.zero_grad()
    accuracy = calculation_accuracy(validation_data, validation_label, model)
print(f"Accuracy of Adam Gradient Descent:{accuracy:.2f}%")

'''
Accuracy of Gradient Descent:90.46%
Accuracy of Dynamic Gradient Descent:90.20%
Accuracy of Adam Gradient Descent:94.06%
'''
