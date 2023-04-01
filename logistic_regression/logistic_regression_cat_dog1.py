import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


np.random.seed(100)
dogs = np.array([[8.9, 12], [9, 11], [10, 13], [9.9, 11.2], [12.2, 10.1], [9.8, 13], [8.8, 11.2]], dtype=np.float32)
cats = np.array([[3, 4], [5, 6], [3.5, 5.5], [4.5, 5.1], [3.4, 4.1], [4.1, 5.2], [4.4, 4.4]], dtype=np.float32)
y = np.array([0] * 7 + [1] * 7, np.int32).reshape(-1, 1)
X = np.vstack((dogs, cats))

b = 0
w = np.random.normal(0, 1, size=(2, 1))
lr = 0.09
epoch = 10000
batch_size = X.shape[0]

for e in range(epoch):
    Y = X @ w + b
    PY = sigmoid(Y)
    loss = -np.mean(y * np.log(PY) + (1 - y) * np.log(1 - PY))
    G = (PY - y) / batch_size
    delta_w = X.T @ G
    delta_b = np.sum(G)
    w = w - delta_w * lr
    b = b - delta_b * lr
    print(loss)

while True:
    f1 = float(input("请输入第一个特征:"))
    f2 = float(input("请输入第二个特征:"))
    x = np.array([f1, f2])

    p = x @ w + b
    p = sigmoid(p)
    if p > 0.5:
        print("猫")
    else:
        print("狗")
