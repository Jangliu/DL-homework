import math
import numpy as np

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = np.array([[0, 1, 1, 0]])
W1 = np.random.rand(3, 4)
W2 = np.random.rand(4, 1)


def Sigmoid(x):
    temp = []
    if type(x) != type(X):
        y = 1 / (1 + math.exp(-x))
    else:
        for element in x.flat:
            temp.append(1 / (1 + math.exp(-element)))
        y = np.array(temp)
    return y


def BackpropXOR(X, D):
    global W1
    global W2
    alpha = 0.9
    for i in range(0, 4):
        d = D[0][i]
        x = X[i]
        v1 = np.dot(x, W1)  # v1 1x4的矩阵
        y1 = Sigmoid(v1)  # y1 1x4的矩阵
        v2 = np.dot(y1, W2)
        y2 = Sigmoid(v2)
        e2 = d - y2
        delta2 = y2 * (1 - y2) * e2
        e1 = np.dot(W2, delta2)  # e1 4x1的矩阵
        delta1 = y1 * (1 - y1) * e1  # delta1 4x1矩阵
        dW1 = alpha * np.reshape(x, (3, 1)) * np.reshape(delta1, (1, 4))
        W1 = W1 + dW1
        dW2 = np.reshape(alpha * delta2 * y1, (4, 1))
        W2 = W2 + dW2


start_time = time.time()
for k in range(0, 10000):
    BackpropXOR(X, D)
for i in range(0, 4):
    d = D[0][i]
    x = X[i]
    v1 = np.dot(x, W1)  # v1 1x4的矩阵
    y1 = Sigmoid(v1)  # y1 1x4的矩阵
    v2 = np.dot(y1, W2)
    y2 = Sigmoid(v2)
    print(y2)
end_time = time.time()
print(end_time - start_time, "s")
