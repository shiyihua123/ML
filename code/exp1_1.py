import copy
import random
import matplotlib.pyplot as plt


import numpy as np

#目标函数f(x)=(x-2)^2
def f_x(x):
    return (x-1)**2
# 导函数
def x_grad(x):
    return 2*x-4
def exp1(alpha):
    # 训练批次
    epoch = 100
    # 学习率
    # alpha = 0.9
    # 初始点位置
    x = -1.5
    x0 = copy.deepcopy(x)
    # 有一系列点的坐标，存储在一个列表中
    points = []
    for e in range(epoch):
        p = (x, f_x(x))
        points.append(p)
        x = x-alpha * x_grad(x)
        # if e % 100 == 0:
        #     print("loss:", f_x(x))

    plt.figure()
    # 画目标函数
    X = np.linspace(-2, 3, 100)
    Y = f_x(X)
    plt.plot(X, Y,  marker='o', markersize=1)

    # 画梯度下降轨迹
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    # 绘制梯度下降的轨迹
    plt.plot(x_values, y_values, marker='o',markersize=1)

    # 设置坐标轴标题等其他属性
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.suptitle('Gradient descent of f(x) = (x-2)^2')
    title = "x0="+str(x0)+'&alpha'+str(alpha)
    plt.title(title)
exp1(0.9)
exp1(0.4)
exp1(0.1)
exp1(0.01)
# 展示图形
plt.show()





