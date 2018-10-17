import math
from numpy import *

W = mat([[0], [0], [0]])
X = mat([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = mat([[0], [0], [1], [1]])


def Sigmod(x):
    y = 1 / (1 + exp(-x))
    return y


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


for i in range(0, 2000):
    Delta_SGDandBatch(X, D)
for i in range(0, 4):
    v = dot(X[i], W)
    y = Sigmod(v)
    print("y = ", y, "\n")
