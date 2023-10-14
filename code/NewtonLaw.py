from random import Random
import numpy as np
from math import *
def hx(x):
    return x**2-2**x

def x_grad(x):
    return 2*x-2**x*log(2)
epochs = 1000
# 牛顿法
x0 = -1
for e in range(epochs):
    x0 = x0-(hx(x0)/x_grad(x0))
print(x0)

# 梯度下降法
# alpha = 1e-1
# x0 = -0.5
# for e in range(epochs):
#     x0 = x0-alpha*x_grad(x0)
# print(x0)


