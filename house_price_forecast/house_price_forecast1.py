# 1. data read
start_year = 2000
end_year = 2022

years = [i for i in range(start_year, end_year + 1)]

prices = [7000, 7100, 7400, 8000, 10000, 12000,
          14000, 12000, 13000, 16000, 20000,
          25000, 32000, 32000, 33000, 37000, 40000,
          50000, 55000, 60000, 64000, 70000, 73000]

# 2. data preprocessing
x = years
y = prices

# 3. hyperparameter initialization
epoch = 10
lr = 2e-07

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
    for i in range(len(y)):
        train_x = x[i]
        train_y = y[i]

        predict_y = k * train_x + b  # forward

        loss = (predict_y - train_y) ** 2  # loss

        delta_k = 2 * (predict_y - train_y) * train_x  # backward
        delta_b = 2 * (predict_y - train_y)

        k = k - delta_k * lr  # update model parameters
        b = b - delta_b * lr

        # validation...

        print("loss:{:.2f}".format(loss))  # output

# 7. output test set metrics (batch test, check model performance)...

# 8. model deployment, open prediction interface (streaming, single, for users)...
