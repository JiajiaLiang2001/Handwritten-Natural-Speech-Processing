import numpy as np


def sigmoid(x):
    return np.clip(1 / (1 + np.exp(-x)), 1e-10, 0.999999)


np.random.seed(100)
dogs = np.array([[8.9, 12], [9, 11], [10, 13], [9.9, 11.2], [12.2, 10.1], [9.8, 13], [8.8, 11.2]], dtype=np.float32)
cats = np.array([[3, 4], [5, 6], [3.5, 5.5], [4.5, 5.1], [3.4, 4.1], [4.1, 5.2], [4.4, 4.4]], dtype=np.float32)
y = np.array([0] * 7 + [1] * 7, np.int32).reshape(-1, 1)
X = np.vstack((dogs, cats))

b1 = 0
b2 = 0
w1 = np.random.normal(0, 1, size=(2, 50))
w2 = np.random.normal(0, 1, size=(50, 1))
lr = 0.005
epoch = 10000
batch_size = X.shape[0]

for e in range(epoch):
    H = X @ w1 + b1  # 1
    Y = H @ w2 + b2  # 2
    # 3
    PY = sigmoid(Y)
    loss = -np.mean(y * np.log(PY) + (1 - y) * np.log(1 - PY))

    G2 = PY - y
    delta_w2 = H.T @ G2  # 5
    delta_b2 = np.mean(G2)

    G1 = G2 @ w2.T  # 6
    delta_w1 = X.T @ G1  # 7
    delta_b1 = np.mean(G1)

    w1 = w1 - delta_w1 * lr
    w2 = w2 - delta_w2 * lr
    b1 = b1 - delta_b1
    b2 = b2 - delta_b2
    print(loss)

while True:
    f1 = float(input("请输入第一个特征:"))
    f2 = float(input("请输入第二个特征:"))
    x = np.array([f1, f2])
    h = x @ w1 + b1
    y = h @ w2 + b2
    p = sigmoid(y)
    if p > 0.5:
        print("猫")
    else:
        print("狗")
