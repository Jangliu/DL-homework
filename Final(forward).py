import os
import struct
import numpy as np
from scipy import signal
import tensorflow as tf

path = "D:\\work\\MINST\\"
W1 = np.random.randn(20, 9, 9)


def ReLU(x):
    temp = []
    Size = x.shape
    for element in x.flat:
        temp.append(max(0, element))
    y = np.array(temp)
    y = y.reshape(Size)
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


def load_mnist(path, kind):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(
            imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

if __name__ == "__main__":
    images, labels = load_mnist(path, "train")
    X = []
    for i in range(len(images)):
        Sum = sum(images[i])
        temp = images[i]/Sum
        X.append(temp.reshape(28, 28))
    for i in range(len(X)):
        V1 = []
        x = X[i]
        for w1 in W1:
            V1.append(signal.convolve2d(w1, np.rot90(x, 2), mode='valid'))
        Y1 = []
        for v1 in V1:
            Y1.append(ReLU(v1))
        n = input()
    
    
