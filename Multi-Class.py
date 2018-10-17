import numpy as np

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
D = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
W1 = np.random.rand(25, 40)
W2 = np.random.rand(40, 5)


def Sigmoid(x):
    temp = []
    if type(x) != type(X):
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
    temp = []
    for element in x.flat:
        if element >= 700:
            y = np.zeros(1, 5)
            return y
        elif element <= -700:
            sum_ex += 0
        else:
            sum_ex += np.exp(element)
    for i in range(x.size):
        temp.append(np.exp(x[i]) / sum_ex) 
    y = np.array(temp)
    return y


def Multi_Class(X, D):
    global W1
    global W2
    alpha = 0.9
    for i in range(0, 5):
        x = X[i]
        d = D[i]
        v1 = np.dot(x, W1)  # v1 1x50的矩阵
        y1 = Sigmoid(v1)  # y1 1x50的矩阵
        v2 = np.dot(y1, W2)  # V2 1x5矩阵
        y2 = Sigmoid(v2)
        e2 = d - y2
        delta2 = e2
        e1 = np.dot(W2, delta2)  # e1 1x50的矩阵
        delta1 = y1 * (1 - y1) * e1  # delta1 1x50矩阵
        dW1 = alpha * np.reshape(x, (25, 1)) * np.reshape(delta1, (1, 40))
        W1 = W1 + dW1
        y1 = np.reshape(40, 1)
        delta2 = delta2.T
        dW2 = alpha * y1 * delta2
        W2 = W2 + dW2


for k in range(0, 10000):
    Multi_Class(X, D)
for i in range(0, 5):
    x = X[i]
    v1 = np.dot(x, W1)
    y1 = Sigmoid(v1)
    v2 = np.dot(y1, W2)
    y2 = Softmax(v2)
    print(y2)
