import numpy as np
import math
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
W1 = 2*np.random.rand(25, 20)-1
W2 = 2*np.random.rand(20, 20)-1
W3 = 2*np.random.rand(20, 20)-1
W4 = 2*np.random.rand(20, 5)-1


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


def Dropout(x, ratio):
    temp = []
    n = x.shape
    x = x.reshape(1, n[0])
    m, n = x.shape
    xm = np.zeros((m, n))
    num = round(m*n*(1-ratio))
    randlist = np.random.rand(m, n).tolist()
    sortedlist = sorted(randlist[0])
    for i in range(0, num):
        temp.append(sortedlist.index(randlist[0][i]))
    for k in temp:
        xm[0][k] = m*n/num
    y = xm
    return y


def DeepDropout(X, D):
    global W1
    global W2
    global W3
    global W4
    alpha = 0.1
    for i in range(0, 5):
        x = X[i].reshape(1, 25)
        d = D[i]
        v1 = np.dot(x, W1)
        y1 = Sigmoid(v1)
        y1 = y1*Dropout(y1, 0.2)
        v2 = np.dot(y1, W2)
        y2 = Sigmoid(v2)
        y2 = y2*Dropout(y2, 0.2)
        v3 = np.dot(y2, W3)
        y3 = Sigmoid(v3)
        y3 = y3*Dropout(y3, 0.2)
        v4 = np.dot(y3, W4)
        y4 = Softmax(v4)
        e4 = d-y4
        delta4 = e4
        e3 = np.dot(delta4, W4.T)
        delta3 = y3*(1-y3)*e3
        e2 = np.dot(delta3, W3.T)
        delta2 = y2*(1-y2)*e2
        e1 = np.dot(delta2, W2.T)
        delta1 = y1*(1-y1)*e1
        W1 = W1+alpha*np.dot(x.T, delta1.reshape(1, 20))
        W2 = W2+alpha*np.dot(y1.reshape(20, 1), delta2.reshape(1, 20))
        W3 = W3+alpha*np.dot(y2.reshape(20, 1), delta3.reshape(1, 20))
        W4 = W4+alpha*np.dot(y3.reshape(20, 1), delta4.reshape(1, 5))


for k in range(0, 20000):
    DeepDropout(X, D)
print("results of train data input:\n")
for i in range(0, 5):
    x = X[i]
    d = D[i]
    v1 = np.dot(x, W1)
    y1 = Sigmoid(v1)
    v2 = np.dot(y1, W2)
    y2 = Sigmoid(v2)
    v3 = np.dot(y2, W3)
    y3 = Sigmoid(v3)
    v4 = np.dot(y3, W4)
    y4 = Softmax(v4)
    print(y4)
print("results of test data input:\n")
for i in range(0, 5):
    x = Test[i]
    d = D[i]
    v1 = np.dot(x, W1)
    y1 = Sigmoid(v1)
    v2 = np.dot(y1, W2)
    y2 = Sigmoid(v2)
    v3 = np.dot(y2, W3)
    y3 = Sigmoid(v3)
    v4 = np.dot(y3, W4)
    y4 = Softmax(v4)
    print(y4)
