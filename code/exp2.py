import copy
import random
import numpy as np
import matplotlib.pyplot as plt

# 真实的模型
def h_x(X):
    return 2*X[0]+1*X[1]+5*X[2]

# 批量梯度下降法
def train1(Theta,epochs,alpha,X,Y):
    print()
    print((' ========== LOSS||Batch Gradient Descent ========== '))
    # 画损失函数图像时的步长
    x_plt = []
    loss_array = []
    for e in range(1, epochs + 1):
        loss = 0
        # ============================批量梯度下降法，每次使用全部样本来更新参数================

        grad = np.array([0, 0, 0])

        for i in range(len(X)):
            # 模型的出来的结果
            h = (Theta*np.mat(X[i]).T).item()
            # 计算模型得到的结果与真实结果Y之间的损失值
            loss += (h - Y[i]) ** 2
            # 更新参数，梯度下降法
            grad = grad + (h - Y[i]) * X[i]

        Theta = Theta - alpha * grad

        # =============================================================================
        loss *= 0.5
        loss /= len(X)


        # if e % (epochs/200) == 0:
        x_plt.append(e)
        loss_array.append(loss)
        # print("------------------------epoch=", e)
        # print("loss=", loss)
        # print("θ0=", Theta[0], "||θ1=", Theta[1], "||θ2=", Theta[2])
            # θ0=2,θ1=1,θ2=5
        if e==epochs-1:
            print("loss=", loss)
    print("θ0=", Theta[0], "||θ1=", Theta[1], "||θ2=", Theta[2])
    y_plt = loss_array
    # plt.subplot(3, 1, 1)
    plt.figure()
    plt.plot(x_plt, y_plt, marker='o', markersize=1)
    plt.title('LOSS||Batch Gradient Descent')
    # 显示网格线
    plt.grid(True)
# 随机梯度下降法
def train2(Theta,epochs,alpha,X,Y):
    print()
    print(' ========== LOSS||Stochastic Gradient Descent ========== ')
    # 画损失函数图像时的步长
    x_plt = []
    loss_array = []
    for e in range(1, epochs + 1):
        loss = 0
        #======================= 随机梯度下降法，即每次只用一个样本来更新参数===================
        # i = random.randint(0,len(X)-1)
        for i in range(len(X)):
            # 模型的出来的结果
            # h = Theta[0] * X[i][0] + Theta[1] * X[i][1] + Theta[2] * X[i][2]
            h = (Theta*np.mat(X[i]).T).item()
            # 计算模型得到的结果与真实结果Y之间的损失值
            loss += (h - Y[i]) ** 2
            # 更新参数，梯度下降法
            Theta = Theta - alpha * (h - Y[i]) * X[i]
            # Theta[1] = Theta[1] - alpha * (h - Y[i]) * X[i][1]
            # Theta[2] = Theta[2] - alpha * (h - Y[i]) * X[i][2]
        # =============================================================================

        loss *= 0.5

        loss/=len(X)

        # if e % (epochs/200) == 0:
        x_plt.append(e)
        loss_array.append(loss)
        # print("------------------------epoch=", e)
        # print("loss=", loss)
        # print("θ0=", Theta[0], "||θ1=", Theta[1], "||θ2=", Theta[2])

        if e==epochs-1:
            print("loss=", loss)
    print("θ0=", Theta[0], "||θ1=", Theta[1], "||θ2=", Theta[2])
    y_plt = loss_array
    # plt.subplot(3, 1, 2)
    plt.figure()
    plt.plot(x_plt, y_plt, marker='o', markersize=1)
    plt.title('LOSS||Stochastic Gradient Descent')
    # 显示网格线
    plt.grid(True)
# 小批量梯度下降法
def train3(Theta,epochs,alpha,X,Y,batch):
    print()
    print(' ========== LOSS||Mini-Batch Gradient Descent ========== ')
    # 画损失函数图像时的步长
    x_plt = []
    loss_array = []
    for e in range(1, epochs + 1):
        loss = 0
        #======================= 小批量梯度下降法，即每次只用一组样本来更新参数===================
        batch = batch
        LB = random.randint(0, len(X)-batch)

        grad = np.array([0, 0, 0])


        for i in range(LB, LB+batch):
            # 模型的出来的结果
            h = (Theta*np.mat(X[i]).T).item()
            # 计算模型得到的结果与真实结果Y之间的损失值
            loss += (h - Y[i]) ** 2
            # 更新参数，梯度下降法
            grad = grad + (h - Y[i]) * X[i]

        Theta = Theta - alpha * grad

        # =============================================================================

        loss *= 0.5
        loss /= batch


        # if e % (epochs/200) == 0:
        x_plt.append(e)
        loss_array.append(loss)
        # print("------------------------epoch=", e)
        # print("loss=", loss)
        # print("θ0=", Theta[0], "||θ1=", Theta[1], "||θ2=", Theta[2])

        if e==epochs-1:
            print("loss=", loss)
    print("θ0=", Theta[0], "||θ1=", Theta[1], "||θ2=", Theta[2])
    y_plt = loss_array
    # plt.subplot(3, 1, 3)
    plt.figure()
    plt.plot(x_plt, y_plt, marker='o', markersize=1)
    plt.title('LOSS||Mini-Batch Gradient Descent')
    # 显示网格线
    plt.grid(True)



if __name__ == '__main__':
    # 生成数据
    X_data = []
    Y_data = []
    mean = 0  # 均值为0
    std_deviation = 1  # 标准差为0
    simple_num = 200  # 样本数量

    for i in range(simple_num):
        x = np.array([1, random.uniform(-10, 10), random.uniform(-10, 10)])
        X_data.append(x)
        # 高斯扰动：np.random.randn() * std_deviation + mean
        Y_data.append(h_x(x))  # 无扰动
        # Y.append(h_x(x)+np.random.randn() * std_deviation + mean)  #有扰动

    # 使用zip函数将X和Y打包成元组的列表
    data = list(zip(X_data, Y_data))
    # 打乱数据
    random.shuffle(data)
    # 将打乱后的数据重新拆分成X和Y列表
    X_data, Y_data = zip(*data)
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # print(X.shape)
    # print(Y.shape)
    # 随机初始参数θ
    Theta = np.array([random.uniform(-10,10),random.uniform(-10,10),random.uniform(-10,10)])
    # 学习率
    alpha = 1e-4
    # 训练批次
    epochs = 500
    # 真实的θ
    print("θ0=2||θ1=1||θ2=5")
    # 批量梯度下降法
    # train1(copy.deepcopy(Theta),epochs,alpha,X_data,Y_data)
    # 随机梯度下降法
    # train2(copy.deepcopy(Theta),epochs,alpha,X_data,Y_data)
    # 小批量梯度下降法
    # train3(copy.deepcopy(Theta),epochs,alpha,X_data,Y_data,batch=50)


    # print("213")


    plt.show()

