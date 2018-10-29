import math
from numpy import *
import time

W = mat([[0], [0], [0]])
X = mat([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = mat([[0], [0], [1], [1]])


def Sigmod(x):
    y = 1 / (1 + exp(-x))
    return y


def DeltaSGD(X, D):
    global W
    alpha = 0.9
    for i in range(0, 4):
        x = X[i]
        d = D[i]
        v = dot(x, W)
        y = Sigmod(v)
        e = d - y
        delta = y * (1 - y) * e
        dw = alpha * delta * x
        W = W + dw.T


start_time = time.time()
for i in range(0, 1000):
    DeltaSGD(X, D)
for i in range(0, 4):
    v = dot(X[i], W)
    y = Sigmod(v)
    print("y = ", y, "\n")
end_time = time.time()
print(end_time - start_time, "s")
