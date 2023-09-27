import copy
from math import *
import numpy as np
import matplotlib.pyplot as plt

# 构建多项式拟合
def hx(x, Theta):
    x = x.T # 进行完次方后，转置回n*1000，为了跟n个参数θ进行矩阵运算
    # 返回类似θ.T*X的结果，1*n的θ行向量与n*1000的X进行矩阵运算得到1*1000的h(x)
    return np.dot(Theta, x)

if __name__ == '__main__':
    # training set
    X = np.linspace(0, 2*pi, 1000) # 1*1000
    np.random.shuffle(X)
    X_test = copy.deepcopy(X)
    Y = np.sin(X) # 1*1000
    # Y = 1+X+X**2+X**3  # 1*1000

    # 次数
    n = 3
    n += 1
    # 训练批次
    epochs = 1000000
    # 随机初始参数θ
    Theta = np.zeros([1, n])
    # 学习率
    alpha = 2e-7

    # 次数
    exponents = np.array([i for i in range(n)]) # 1*n
    exponents = np.tile(exponents, (len(X), 1)) # 1000*n
    # x
    X = np.tile(X, (n, 1)).T # n * 1000 转置 => 1000*n为了匹配次数
    X = np.power(X, exponents)  # 进行次方 1000*n

    for e in range(1, epochs+1):
        loss = 0.5*np.sum((hx(X, Theta) - Y) ** 2)
        if e % 100000 == 0:
            print("epoch:",e ,"||loss:",loss)
            # print(Theta)

        Theta_grad = np.dot((hx(X, Theta)-Y), X)
        Theta = Theta - alpha * Theta_grad

    plt.figure()
    plt.scatter(X_test, hx(X, Theta))
    plt.show()