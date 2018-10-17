import math
from numpy import *
import time
import matplotlib.pyplot as plt

W = mat([[0], [0], [0]])
X = mat([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = mat([[0], [0], [1], [1]])


def Sigmod(x):
    y = 1 / (1 + exp(-x))
    return y


def DeltaSGD(X, D):
    global W
    alpha = 0.9
    N = 4
    for i in range(0, 4):
        x = X[i]
        d = D[i]
        v = dot(x, W)
        y = Sigmod(v)
        e = d - y
        delta = y * (1 - y) * e
        dw = alpha * delta * x
        W = W + dw.T


def Delta_SGDandBatch(X, D):
    global W
    alpha = 0.8
    N = 4
    dw = mat([[0.0], [0.0], [0.0]])
    for i in range(0, 2):
        x = X[i]
        d = D[i]
        v = dot(x, W)
        y = Sigmod(v)
        e = d - y
        delta = y * (1 - y) * e
        dw += (alpha * delta * x).T
    W = W + dw / 2
    for i in range(2, 4):
        x = X[i]
        d = D[i]
        v = dot(x, W)
        y = Sigmod(v)
        e = d - y
        delta = y * (1 - y) * e
        dw += (alpha * delta * x).T
    W = W + dw / 2


def DeltaBatch(X, D):
    global W
    alpha = 0.8
    N = 4
    dw = mat([[0.0], [0.0], [0.0]])
    for i in range(0, 4):
        x = X[i]
        d = D[i]
        v = dot(x, W)
        y = Sigmod(v)
        e = d - y
        delta = y * (1 - y) * e
        dw += (alpha * delta * x).T
    W = W + dw / 4


def main():
    global W
    Deviation_SGD = []
    Deviation_Batch = []
    Deviation_SGDandBatch = []
    t1 = time.time()
    for i in range(0, 1000):
        DeltaSGD(X, D)
    t2 = time.time()
    print("SGD:", t2 - t1, "s")
    W = mat([[0], [0], [0]])
    t3 = time.time()
    for i in range(0, 1000):
        DeltaBatch(X, D)
    t4 = time.time()
    print("Batch:", t4 - t3, "s")
    W = mat([[0], [0], [0]])
    t5 = time.time()
    for i in range(0, 1000):
        Delta_SGDandBatch(X, D)
    t6 = time.time()
    print("SGD+Batch:", t6 - t5, "s")
    W = mat([[0], [0], [0]])
    for i in range(0, 1000):
        DeltaSGD(X, D)
        deviation = 0
        for j in range(0, 4):
            v = dot(X[j], W)
            deviation += (D[j] - Sigmod(v))**2
        temp = deviation.tolist()
        Deviation_SGD.append(temp[0][0])
    W = mat([[0], [0], [0]])
    for i in range(0, 1000):
        DeltaBatch(X, D)
        deviation = 0
        for j in range(0, 4):
            v = dot(X[j], W)
            deviation += (D[j] - Sigmod(v))**2
        temp = deviation.tolist()
        Deviation_Batch.append(temp[0][0])
    W = mat([[0], [0], [0]])
    for i in range(0, 1000):
        Delta_SGDandBatch(X, D)
        deviation = 0
        for j in range(0, 4):
            v = dot(X[j], W)
            deviation += (D[j] - Sigmod(v))**2
        temp = deviation.tolist()
        Deviation_SGDandBatch.append(temp[0][0])
    plt.plot(Deviation_SGD, label="SGD")
    plt.plot(Deviation_Batch, label="Batch")
    plt.plot(Deviation_SGDandBatch, label="SGD+Batch")
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc=3,
        ncol=3,
        mode='expand',
        borderaxespad=0.)
    plt.show()
    n = input()


main()
