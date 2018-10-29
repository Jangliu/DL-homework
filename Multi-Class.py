import numpy as np
import math
# numpy矩阵正常相乘为用dot，点乘用*
X = np.array([
    [
        0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
        0
    ],
    [
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
        1
    ],
    [
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        0
    ],
    [
        0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        0
    ],
    [
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        0
    ],
])
Test = np.array([[0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                     1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                     1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                 [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
                     1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1,
                     1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]

                 ])
D = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
W1 = 2*np.random.rand(25, 50)-1
W2 = 2*np.random.rand(50, 5)-1


def Sigmoid(x):
    temp = []
    if isinstance(x, type(np.mat(X))):
        y = 1 / (1 + np.exp(-x))
    else:
        for element in x.flat:
            if element >= 700:
                temp.append(0)
            elif element <= -700:
                temp.append(1)
            else:
                temp.append(1 / (1 + np.exp(-element)))
        y = np.array(temp)
    return y


def Softmax(x):
    sum_ex = 0
    y = []
    for element in x.flat:
        sum_ex += math.exp(element)
    for element in x.flat:
        y.append(math.exp(element)/sum_ex)
    y = np.array(y)
    return y


def Multi_Class(X, D):
    global W1
    global W2
    alpha = 0.9
    for i in range(0, 5):
        x = X[i].reshape(1, 25)
        d = D[i]
        v1 = np.dot(x, W1)  # v1 1x50的矩阵
        y1 = Sigmoid(v1)  # y1 1x50的矩阵
        v2 = np.dot(y1, W2)  # V2 1x5矩阵
        y2 = Softmax(v2)
        e2 = d - y2
        delta2 = e2
        e1 = np.dot(delta2, W2.T)  # e1 1x50的矩阵
        delta1 = y1*(1-y1)*e1  # delta1 1x50矩阵
        dW1 = alpha * np.dot(x.T, delta1.reshape(1, 50))
        W1 = W1 + dW1
        dW2 = alpha * np.dot(y1.reshape(50, 1), delta2.reshape(1, 5))
        W2 = W2 + dW2


for k in range(0, 10000):
    Multi_Class(X, D)
print("Trained results:\n")
for i in range(0, 5):
    x = X[i]
    v1 = np.dot(x, W1)
    y1 = Sigmoid(v1)
    v2 = np.dot(y1, W2)
    y2 = Softmax(v2)
    print(y2)
print("Test:\n")
for i in range(0, 5):
    x = Test[i]
    v1 = np.dot(x, W1)
    y1 = Sigmoid(v1)
    v2 = np.dot(y1, W2)
    y2 = Softmax(v2)
    print(y2)
