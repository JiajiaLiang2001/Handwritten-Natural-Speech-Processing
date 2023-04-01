"""
x ** 2 = 3

--> x = ?
"""

from math import sqrt

label = 3

x = 4

epoch = 100
lr = 0.01

for e in range(epoch):
    pre = x ** 2

    loss = (pre - label) ** 2

    delta_x = 2 * (pre - label) * 2 * x

    x = x - delta_x * lr

    print(x)

print("\nPredicted value:\t{}".format(x))
print("\nActual value:\t{}\n".format(sqrt(label)))
