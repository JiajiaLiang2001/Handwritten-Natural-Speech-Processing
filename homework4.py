import os

import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler as mms
from dataset_dataloader.dataset6_dataloader import MyDataset, MyDataLoader

file_path = os.path.join("data", "homework4", "housing.csv")
data = read_csv(file_path)

# print(data.head(5))
#       CRIM    ZN  INDUS  CHAS    NOX  ...    TAX  PIRATIO       B  LSTAT  MEDV
# 0  0.00632  18.0   2.31     0  0.538  ...  296.0     15.3  396.90   4.98  24.0
# 1  0.02731   0.0   7.07     0  0.469  ...  242.0     17.8  396.90   9.14  21.6
# 2  0.02729   0.0   7.07     0  0.469  ...  242.0     17.8  392.83   4.03  34.7
# 3  0.03237   0.0   2.18     0  0.458  ...  222.0     18.7  394.63   2.94  33.4
# 4  0.06905   0.0   2.18     0  0.458  ...  222.0     18.7  396.90   5.33  36.2
#
# [5 rows x 14 columns]

X, y = data[data.columns.delete(-1)].values, data['MEDV'].values.reshape(-1,1)

X_scaler = mms()
X_scaler.fit(X)
X = X_scaler.transform(X)

y_scaler = mms()
y_scaler.fit(y)
y = y_scaler.transform(y)

sample_size = X.shape[0]
feature_size = X.shape[1]

w = np.random.rand(feature_size, 1)  # standard normal distribution
b = float(np.random.rand(1))
lr = 0.00001
epoch = 30
batch_size = 20

dataset = MyDataset(X, y, batch_size=batch_size, shuffle=True)

for e in range(epoch):
    for train_X, train_y in dataset:
        predict_y = train_X @ w + b
        loss = np.mean((predict_y - train_y) ** 2)

        G = 2 * (predict_y - train_y)

        delta_w = train_X.T @ G
        delta_b = np.mean(G)

        w = w - delta_w * lr
        b = b - delta_b * lr

        print("loss:{}".format(loss))



