from sklearn.utils import shuffle
import numpy as np 

__all__ = ["train_test_split"]

def train_test_split(X, y, percent = .15):
    X, y = shuffle(X, y)
    split_index = np.int(np.round(X.shape[0])*.15)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[-split_index:], y[-split_index:]
    return X_train, y_train, X_test, y_test