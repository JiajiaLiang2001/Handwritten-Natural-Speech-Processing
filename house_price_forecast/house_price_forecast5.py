# 1. data read
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as mms

file_path = os.path.join("..", "data", "house_price_forecast", "上海二手房价.csv")
data = pd.read_csv(file_path)
prices = data["房价（元/平米）"].values.reshape(-1, 1)
data_shape = prices.shape
floors = data["楼层"].values.reshape(data_shape)
years = data["建成年份"].values.reshape(data_shape)

# 2. data preprocessing
prices_scaler = mms()
prices_scaler.fit(prices)
prices = prices_scaler.transform(prices)

floors_scaler = mms()
floors_scaler.fit(floors)
floors = floors_scaler.transform(floors)

years_scaler = mms()
years_scaler.fit(years)
years = years_scaler.transform(years)

y = prices
x1 = floors
x2 = years

# 3. hyperparameter initialization
epoch = 10
lr = 0.001
batch_size = 4
batch_num = int(np.ceil(len(data_shape) / batch_size))

# 4. model parameter initialization
k1 = 0
k2 = 0
b = 0

# 5. model
"""
y = k1 * x_1 + k2 * x_2 + b:

    - x_1:floors
    - x_2:years
    - y:prices
"""

# 6. train
for e in range(epoch):
    for bt in range(batch_num):
        batch_start = bt * batch_size
        batch_end = (bt + 1) * batch_size
        if batch_end > data_shape[0]: batch_end = data_shape[0]
        batch_index = np.arange(batch_start, batch_end, dtype=int)
        train_x1 = x1[batch_index]
        train_x2 = x2[batch_index]
        train_y = y[batch_index]

        predict_y = k1 * train_x1 + k2 * train_x2 + b  # forward

        loss = np.mean((predict_y - train_y) ** 2)  # loss

        delta_k1 = np.mean(2 * (predict_y - train_y) * train_x1)  # backward
        delta_k2 = np.mean(2 * (predict_y - train_y) * train_x2)
        delta_b = np.mean(2 * (predict_y - train_y))

        k1 = k1 - delta_k1 * lr  # update model parameters
        k2 = k2 - delta_k2 * lr
        b = b - delta_b * lr

        # validation...

        print("loss:{}".format(loss))  # output

# 7. output test set metrics (batch test, check model performance)
while True:
    org_input_floor = np.array(input("Please enter the floor to be forecasted:")).reshape(1, 1)
    org_input_year = np.array(input("Please enter the year to be forecasted:")).reshape(1, 1)

    input_floor = floors_scaler.transform(org_input_floor)
    input_year = years_scaler.transform(org_input_year)

    output_price = input_floor * k1 + input_year * k2 + b

    org_output_price = prices_scaler.inverse_transform(output_price)

    print("Prices for {} are:{:.2f}".format(org_input_year[0][0], org_output_price[0][0]))

# 8. model deployment, open prediction interface (streaming, single, for users)...
