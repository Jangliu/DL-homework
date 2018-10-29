import math
import numpy as np
import matplotlib.pyplot as plt
import time

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = np.array([[0, 1, 1, 0]])
W1 = np.random.rand(3, 4)
W2 = np.random.rand(4, 1)
W1_BP = W1
W2_BP = W2
W1_CE = W1
W2_CE = W2


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
    global W1_BP
    global W2_BP
    alpha = 0.9
    for i in range(0, 4):
        d = D[0][i]
        x = X[i]
        v1 = np.dot(x, W1_BP)  # v1 1x4的矩阵
        y1 = Sigmoid(v1)  # y1 1x4的矩阵
        v2 = np.dot(y1, W2_BP)
        y2 = Sigmoid(v2)
        e2 = d - y2
        delta2 = y2 * (1 - y2) * e2
        e1 = np.dot(W2_BP, delta2)  # e1 4x1的矩阵
        delta1 = y1 * (1 - y1) * e1  # delta1 4x1矩阵
        dW1 = alpha * np.reshape(x, (3, 1)) * np.reshape(delta1, (1, 4))
        W1_BP = W1_BP + dW1
        dW2 = np.reshape(alpha * delta2 * y1, (4, 1))
        W2_BP = W2_BP + dW2


def BackpropXOR_CE(X, D):
    global W1_CE
    global W2_CE
    alpha = 0.9
    for i in range(0, 4):
        d = D[0][i]
        x = X[i]
        v1 = np.dot(x, W1_CE)  # v1 1x4的矩阵
        y1 = Sigmoid(v1)  # y1 1x4的矩阵
        v2 = np.dot(y1, W2_CE)
        y2 = Sigmoid(v2)
        e2 = d - y2
        delta2 = e2
        e1 = np.dot(W2_CE, delta2)  # e1 4x1的矩阵
        delta1 = y1 * (1 - y1) * e1  # delta1 4x1矩阵
        dW1 = alpha * np.reshape(x, (3, 1)) * np.reshape(delta1, (1, 4))
        W1_CE = W1_CE + dW1
        dW2 = np.reshape(alpha * delta2 * y1, (4, 1))
        W2_CE = W2_CE + dW2


start_time = time.time()
Deviation_BP = []
Deviation_CE = []
for k in range(0, 10000):
    BackpropXOR(X, D)
    deviation = 0
    for i in range(0, 4):
        d = D[0][i]
        x = X[i]
        v1 = np.dot(x, W1_BP)  # v1 1x4的矩阵
        y1 = Sigmoid(v1)  # y1 1x4的矩阵
        v2 = np.dot(y1, W2_BP)
        y2 = Sigmoid(v2)
        deviation += (d - Sigmoid(y2))**2
        temp = deviation.tolist()
        Deviation_BP.append(temp[0])
for k in range(0, 10000):
    deviation = 0
    BackpropXOR_CE(X, D)
    for i in range(0, 4):
        d = D[0][i]
        x = X[i]
        v1 = np.dot(x, W1_CE)  # v1 1x4的矩阵
        y1 = Sigmoid(v1)  # y1 1x4的矩阵
        v2 = np.dot(y1, W2_CE)
        y2 = Sigmoid(v2)
        deviation += (d - Sigmoid(y2))**2
        temp = deviation.tolist()
        Deviation_CE.append(temp[0])
print("BP results:\n")
for i in range(0, 4):
    d = D[0][i]
    x = X[i]
    v1 = np.dot(x, W1_BP)  # v1 1x4的矩阵
    y1 = Sigmoid(v1)  # y1 1x4的矩阵
    v2 = np.dot(y1, W2_BP)
    y2 = Sigmoid(v2)
    print(y2)
print("CE resulat:\n")
for i in range(0, 4):
    d = D[0][i]
    x = X[i]
    v1 = np.dot(x, W1_CE)  # v1 1x4的矩阵
    y1 = Sigmoid(v1)  # y1 1x4的矩阵
    v2 = np.dot(y1, W2_CE)
    y2 = Sigmoid(v2)
    print(y2)
end_time = time.time()
print(end_time - start_time, "s")
