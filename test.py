import os
import struct
import numpy as np
from scipy import signal
import math
import time
from cProfile import Profile

path = "C:\\Users\\Jangliu\\work\\MINST\\"
W1 = np.random.randn(20, 9, 9)
W3 = (2 * np.random.rand(100, 2000) - 1) / 200
W4 = (2 * np.random.rand(10, 100) - 1) / 100


def ReLU(x):
    temp = []
    Size = x.shape
    for element in x.flat:
        if element > 0:
            temp.append(element)
        else:
            temp.append(0)
    y = np.array(temp)
    y = y.reshape(Size)
    return y


def Softmax(x):
    sum_ex = 0
    y = []
    Max = max(max(x))
    for element in x.flat:
        sum_ex += math.exp(element-Max)
    for element in x.flat:
        y.append(math.exp(element-Max) / sum_ex)
    y = np.array(y)
    return y


def load_mnist(path, kind):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def main():
    global W3
    global W4
    images, labels = load_mnist(path, "train")
    D = []
    for label in labels:
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[label] = 1
        D.append(temp)
    D = np.array(D)
    X = []
    for i in range(len(images)):
        Sum = sum(images[i])
        temp = images[i] / Sum
        X.append(temp.reshape(28, 28))
    for i in range(0, 60000):
        alpha = 0.05
        d = D[i].reshape(10, 1)
        x = X[i]
        V1 = []
        for w1 in W1:
            V1.append(signal.convolve2d(w1, np.rot90(x, 2), mode='valid'))
        Y1 = []
        for v1 in V1:
            Y1.append(ReLU(v1))
        Y2 = []
        for y1 in Y1:
            temp = []
            for j in range(0, 20, 2):
                for k in range(0, 20, 2):
                    k = y1[j:j + 2, k:k + 2]
                    Sum = sum(sum(k))
                    temp.append(Sum / 4)
            temp = np.array(temp)
            Y2.append(temp.reshape(10, 10))
        y2 = Y2[0].reshape(100, 1)
        for j in range(1, len(Y2)):
            y2 = np.vstack((y2, Y2[j].reshape(100, 1)))
        # y2 2000x1
        v3 = np.dot(W3, y2)  # 100x1
        y3 = ReLU(v3)  # 100x1
        v4 = np.dot(W4, y3)
        y4 = Softmax(v4)
        y4 = y4.reshape(10, 1)
        e4 = d - y4  # 10x1
        delta4 = e4  # 10x1
        e3 = np.dot(W4.T, delta4)  # 100x1
        delta3 = y3 * (1 - y3) * e3
        dW3 = alpha * np.dot(delta3, y2.reshape(1, 2000))
        W3 = W3 + dW3
        dW4 = alpha * np.dot(delta4, y3.reshape(1, 100))
        W4 = W4 + dW4


if __name__ == '__main__':
    prof = Profile()
    prof.runcall(main)
    prof.print_stats()
