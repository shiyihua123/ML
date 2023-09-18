#目标函数f(x)=(x-2)^2
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f_x_y(x,y):
    return (x-2)**2+(y-3)**2
# 偏导函数x
def x_grad(x):
    return 2*(x-2)
# 偏导函数x
def y_grad(y):
    return 2*(y-3)

fig = plt.figure()
def exp2(alpha,idx):
    # 训练批次
    epoch = 1000
    # 学习率
    alpha = alpha
    # 初始点位置
    x = -10
    y = -10
    x0 = copy.deepcopy(x)
    y0 = copy.deepcopy(y)
    # 有一系列点的坐标，存储在一个列表中
    points = []
    for e in range(epoch):
        p = (x, y, f_x_y(x, y))
        points.append(p)
        x = x - alpha * x_grad(x)
        y = y - alpha * y_grad(y)
        # if e % 100 == 0:
        # print("x:",x,"y:",y,"loss:", f_x_y2(x,y))
    # print(points)
    # plt.subplot(1, 4, idx)

    # # 画目标函数
    X = np.linspace(-10, 10, 50)
    Y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(X, Y)
    Z = f_x_y(X, Y)

    ax = fig.add_subplot(idx, projection=Axes3D.name)
    # # 画梯度下降轨迹
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    z_values = [point[2] for point in points]

    # # 绘制梯度下降的轨迹
    ax.plot(x_values, y_values, z_values, color='r', marker='o', markersize=1)

    ax.plot_surface(X, Y, Z, cmap="rainbow", alpha=0.6)

    title = "x0="+str(x0)+"&y0="+str(y0)+'&alpha'+str(alpha)
    ax.set_title(title,fontsize = 7)


idx = 221
exp2(0.9,idx)
idx+=1
exp2(0.4,idx)
idx+=1
exp2(0.1,idx)
idx+=1
exp2(0.01,idx)
plt.show()
