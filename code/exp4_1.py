import random
import pandas as pd
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def generate_data(num_feature, simple_num):
    # 指定生成1000数据样本，以及10特征，外加一个数值为1的x0，跟Θ0组在一起
    neg = 0 # 用来判断标签为0的样本数量够不够simple_num的半，下面的pos作用同理
    pos = 0
    dataset = [] # 最后的数据集
    # 生成11个θ，其中一个为θ0，即b
    Theta = np.mat(np.random.uniform(-1, 1, num_feature + 1))
    while 1:
        # 如果两种标签的数据集都够了规定总样本数量的一半就结束生成
        if pos >= simple_num / 2 and neg >= simple_num / 2:
            break
        X = np.random.uniform(-1, 1, num_feature)
        X = np.mat(np.append(X, 1))
        Z = Theta * X.T
        h = sigmoid(Z.item())
        load = False
        y = None
        if h >= 0.5 and pos < simple_num / 2:
            y = 1
            pos += 1
            load = True

        if h < 0.5 and neg < simple_num / 2:
            y = 0
            neg += 1
            load = True
        if load:
            X = np.squeeze(X)

            data = X.tolist()
            data[0].append(y)
            dataset.append(data[0])

    random.shuffle(dataset)
    # 生成excel里的标签栏
    columns_x = ["x0", "label"]
    for i in range(num_feature):
        columns_name = "x" + str(num_feature - i)
        columns_x.insert(0, columns_name)

    columns_Theta = ["θ0"]
    for i in range(num_feature):
        columns_name = "θ" + str(num_feature - i)
        columns_Theta.insert(0, columns_name)

    # list转dataframe
    df = pd.DataFrame(dataset, columns=columns_x)
    df_Theta = pd.DataFrame(Theta, columns=columns_Theta)
    # 保存到本地excel
    df.to_excel("../data/dataset.xlsx", index=False)
    df_Theta.to_excel("../data/dataset_Theta.xlsx", index=False)

def readdata(path):
    df = pd.read_excel(path)
    data_list = df.values.tolist()
    return data_list

def readPram(p1, p2):
    data_set = readdata(p1)
    X = []
    Y = []
    # 输出列表数据
    for row in data_set:
        X.append(row[:-1])
        Y.append(row[-1])
    Theta = readdata(p2)[0]
    return X, Y, Theta

if __name__ == "__main__":

    num_feature = 10
    simple_num = 1000
    # generate_data(num_feature=num_feature, simple_num=simple_num)


    path1 = "../data/dataset.xlsx"
    path2 = "../data/dataset_Theta.xlsx"
    X, Y, Theta_true = readPram(path1, path2)
    # 随机生成θ值
    Theta = np.mat(np.random.uniform(-1, 1, num_feature + 1))
    X = np.mat(X)
    Y = np.mat(Y)
    temp = np.ones([1, simple_num])
    # 训练超参数
    epochs = 1000
    Alpha = 1e-1
    for e in range(1, epochs+1):
        epochs += 1
        h = sigmoid(Theta*X.T)
        loss = np.multiply(Y, np.log(h)) + np.multiply(1-Y, np.log(1-h))
        loss = (loss * temp.T)/simple_num
        if e % 100 == 0:
            # 有时候损失会超级大，因为h会跟接近0，log出来就非常大，以至于警告报错
            print("epoch：", e, "loss：", loss.item())

            # 西塔因为训练出来的时候会带有倍数，所以需要采用归一化来比较，这也是逻辑回归里sigmoid函数导致的原因
            Theta_normalized = Theta / np.linalg.norm(Theta)
            Theta_true_normalized = Theta_true / np.linalg.norm(Theta_true)
            # 计算余弦相似度，用该方法来比较参数时候很类似
            cosine_similarity = np.dot(Theta_normalized, Theta_true_normalized) / (
                        np.linalg.norm(Theta_normalized) * np.linalg.norm(Theta_true_normalized))
            print(cosine_similarity)
            print("-"*100)

        # 用所有训练样本来更新梯度
        Theta = Theta + Alpha * (Y - h) * X
    # 西塔因为训练出来的时候会带有倍数，所以需要采用归一化来比较
    Theta_normalized = Theta / np.linalg.norm(Theta)
    Theta_true_normalized = Theta_true / np.linalg.norm(Theta_true)
    print("训练的θ：", Theta_normalized)
    print("真实的θ：", Theta_true_normalized)











