"""
(x1 - 2) ** 2 + (x2 + 3) ** 2 = 0

--> x1 = ? x2 = ?
"""

label = 0

x1 = 1
x2 = 1

epoch = 10000
lr = 0.01

for e in range(epoch):
    pre = (x1 - 2) ** 2 + (x2 + 3) ** 2

    loss = (pre - label) ** 2

    delta_x1 = 2 * (pre - label) * 2 * (x1 - 2)
    delta_x2 = 2 * (pre - label) * 2 * (x2 + 3)

    x1 = x1 - lr * delta_x1
    x2 = x2 - lr * delta_x2

    if e % 100 == 0:
        print("x1:{},x2:{}\tloss:{}".format(x1, x2, loss))

print("\nPredicted value:\n\tx1:{},x2:{}".format(x1, x2))
print("\nActual value:\n\tx1:{},x2:{}".format(2, -3))
