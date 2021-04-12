"""
checking if I can import mnist
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def load_mnist(data_dir):
    X_train_raw = np.fromfile(os.path.join(data_dir, 'mnist-train-images.dat'), dtype=np.uint8, offset=16)
    X_train = X_train_raw.reshape(-1, 28*28)

    X_test_raw = np.fromfile(os.path.join(data_dir, 'mnist-test-images.dat'), dtype=np.uint8, offset=16)
    X_test = X_test_raw.reshape(-1, 28*28)

    y_train = np.fromfile(os.path.join(data_dir, 'mnist-train-labels.dat'), dtype=np.uint8, offset=8)

    y_test = np.fromfile(os.path.join(data_dir, 'mnist-test-labels.dat'), dtype=np.uint8, offset=8)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_mnist('mnist')
