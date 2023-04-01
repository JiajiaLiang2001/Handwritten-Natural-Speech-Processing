"""
sin(x) = 1/2

--> x = ?
"""

from math import sin, cos, pi

label = 1 / 2

x = 0

epoch = 1000
lr = 0.01

for e in range(epoch):
    pre = sin(x)

    loss = (pre - label) ** 2

    delta_x = cos(x)

    x = x - delta_x * lr

    print(x)

print("\nPredicted value:\t{}".format(x))
