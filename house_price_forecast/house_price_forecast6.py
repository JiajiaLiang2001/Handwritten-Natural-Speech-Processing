# 1. data read
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as mms

file_path = os.path.join("..", "data", "house_price_forecast", "上海二手房价.csv")
data = pd.read_csv(file_path)
prices = data["房价（元/平米）"].values.reshape(-1, 1)
data_shape = prices.shape
squares = data["面积（平米）"].values.reshape(data_shape)
floors = data["楼层"].values.reshape(data_shape)
years = data["建成年份"].values.reshape(data_shape)

# 2. data preprocessing
prices_scaler = mms()
prices_scaler.fit(prices)
prices = prices_scaler.transform(prices)

squares_scaler = mms()
squares_scaler.fit(squares)
squares = squares_scaler.transform(squares)

floors_scaler = mms()
floors_scaler.fit(floors)
floors = floors_scaler.transform(floors)

years_scaler = mms()
years_scaler.fit(years)
years = years_scaler.transform(years)

y = prices
x1 = squares
x2 = floors
x3 = years
X = np.stack((x1, x2, x3), axis=-1).squeeze(axis=1)

# 3. hyperparameter initialization
epoch = 10
lr = 0.002
batch_size = 4
batch_num = int(np.ceil(len(years) / batch_size))

# 4. model parameter initialization
K = np.random.normal(0, 0.5, size=(X.shape[1], 1))
b = 0

# 5. model
"""
y =X @ K  + b:

    - X:squares,floors,years    dim(X)=(270,3)
    - y:prices
"""

# 6. train
for e in range(epoch):
    for bt in range(batch_num):
        batch_start = bt * batch_size
        batch_end = (bt + 1) * batch_size
        if batch_end > data_shape[0]: batch_end = data_shape[0]
        batch_index = np.arange(batch_start, batch_end, dtype=int)
        train_X = X[batch_index]
        train_y = y[batch_index]

        predict_y = train_X @ K + b  # forward

        loss = np.mean((predict_y - train_y) ** 2)  # loss

        G = 2 * (predict_y - train_y)

        delta_K = train_X.T @ G
        delta_b = np.mean(G)

        K = K - delta_K * lr
        b = b - delta_b * lr

        # validation...

        print("loss:{}".format(loss))  # output

# 7. output test set metrics (batch test, check model performance)
while True:
    org_input_square = np.array(input("Please enter the square to be forecasted:")).reshape(1, 1)
    org_input_floor = np.array(input("Please enter the floor to be forecasted:")).reshape(1, 1)
    org_input_year = np.array(input("Please enter the year to be forecasted:")).reshape(1, 1)

    input_square = squares_scaler.transform(org_input_square)
    input_floor = floors_scaler.transform(org_input_floor)
    input_year = years_scaler.transform(org_input_year)

    x = np.stack((input_square, input_floor, input_year), axis=-1).squeeze(axis=1)
    output_price = x @ K + b

    org_output_price = prices_scaler.inverse_transform(output_price)

    print("Prices for {} are:{:.2f}".format(org_input_year[0][0], org_output_price[0][0]))

# 8. model deployment, open prediction interface (streaming, single, for users)...
