# 1. data read
import numpy as np

start_year = 1980
end_year = 2022

years = np.array([i for i in range(start_year, end_year + 1)])

prices = np.array([7000, 7100, 7400, 8000, 10000, 12000,
                   14000, 12000, 13000, 16000, 20000,
                   25000, 32000, 32000, 33000, 37000, 40000,
                   50000, 55000, 60000, 64000, 70000, 73000])

min_price = np.min(prices)
max_price = np.max(prices)

# 2. data preprocessing
x = (years - start_year) / (end_year - start_year)
y = (prices - min_price) / (max_price - min_price)

# 3. hyperparameter initialization
epoch = 10
lr = 0.5

# 4. model parameter initialization
k = 1
b = -1

# 5. model
"""
y = kx + b:

    - x:years
    - y:prices
"""

# 6. train
for e in range(epoch):
    train_x = x
    train_y = y

    predict_y = k * train_x + b  # forward

    loss = np.mean((predict_y - train_y) ** 2)  # loss

    delta_k = np.mean(2 * (predict_y - train_y) * train_x)  # backward
    delta_b = np.mean(2 * (predict_y - train_y))

    k = k - delta_k * lr  # update model parameters
    b = b - delta_b * lr

    # validation...

    print("loss:{}".format(loss))  # output

# 7. output test set metrics (batch test, check model performance)
while True:
    org_input_year = int(input("Please enter the year to be forecasted："))

    input_year = (org_input_year - start_year) / (end_year - start_year)
    output_price = input_year * k + b

    org_output_price = output_price * (max_price - min_price) + min_price

    print("Prices for {} are:{:.2f}".format(org_input_year, org_output_price))

# 8. model deployment, open prediction interface (streaming, single, for users)...
