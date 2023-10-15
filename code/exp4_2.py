from exp4_1 import readPram
import numpy as np




def g(z):
    result = np.where(z < 0, -1, 1)
    return result


if __name__ == "__main__":

    num_feature = 10
    simple_num = 1000
    path1 = "../data/dataset.xlsx"
    path2 = "../data/dataset_Theta.xlsx"

    X, Y, Theta_true = readPram(path1, path2)
    # 随机生成θ值
    Theta = np.mat(np.random.uniform(-1, 1, num_feature + 1))
    X = np.mat(X)
    Y = np.mat(Y)
    Y = 2 * Y - 1
    # 训练超参数
    epochs = 10000
    Alpha = 3e-3
    for e in range(1, epochs + 1):
        h = g(Theta * X.T)
        equal_matrix = np.equal(Y, h)
        Error = np.where(equal_matrix, 0, Y)

        loss = -Error * (Theta * X.T).T

        # 判断两个矩阵是否相等

        # 随机选择一个不相等的位置，将其改为1，其他位置改为0
        index = np.where(equal_matrix == False)
        if len(index[0]) == 0 :
            break
        random_index = np.random.choice(len(index[0]))
        result = np.zeros_like(Y)
        result[index[0][random_index], index[1][random_index]] = Y[index[0][random_index], index[1][random_index]]
        print("epochs：", e, "loss：", loss.item())
        print("-" * 100)
        Theta = Theta + Alpha * result * X

    Theta_normalized = Theta / np.linalg.norm(Theta)
    Theta_true_normalized = Theta_true / np.linalg.norm(Theta_true)
    print("训练的θ：", Theta_normalized)
    print("真实的θ：", Theta_true_normalized)




