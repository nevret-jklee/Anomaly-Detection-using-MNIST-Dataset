import torch
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

class Dataset(object):
    
    def __init__(self, normalize=True):
        print("\nInitializing Dataset...")

        self.normalize = normalize

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train, self.y_train = x_train, y_train
        self.x_test,  self.y_test  = x_test,  y_test

        self.x_train = np.ndarray.astype(self.x_train, np.float32)
        self.x_test  = np.ndarray.astype(self.x_test,  np.float32)

        self.split_dataset() 


    def split_dataset(self):
        x_total = np.append(self.x_train, self.x_test, axis=0)
        y_total = np.append(self.y_train, self.y_test, axis=0)
        
        x_normal,   y_normal = None, None
        x_abnormal, y_abnormal = None, None

        for y_idx, y in enumerate(y_total):
            x_tmp = np.expand_dims(x_total[y_idx], axis=0)
            y_tmp = np.expand_dims(y_total[y_idx], axis=0)

            if (y == 1):       # as normal
                if (x_normal is None):
                    x_normal = x_tmp
                    y_normal = y_tmp
                else:
                    x_normal = np.append(x_normal, x_tmp, axis=0)
                    y_normal = np.append(y_normal, y_tmp, axis=0)
            else:            # as abnormal
                if (x_abnormal is None):
                    x_abnormal = x_tmp
                    y_abnormal = y_tmp
                else:
                    if (x_abnormal.shape[0] < 1000):
                        x_abnormal = np.append(x_abnormal, x_tmp, axis=0)
                        y_abnormal = np.append(y_abnormal, y_tmp, aixs=0)

            if (not(x_normal is None) and not(x_abnormal is None)):
                if ((x_normal.shape[0] >= 2000) and x_abnormal.shape[0] >= 1000):
                    break

        self.x_train, self.y_train = x_normal[:1000], y_normal[:1000]
        self.x_test,  self.y_test  = x_normal[1000:], y_normal[1000:]
        self.x_test = np.append(self.x_test, x_abnormal, axis=0)
        self.y_test = np.append(self.y_test, y_abnormal, axis=0)

    
    def reset_idx(self):
        self.idx_train, self.idx_test = 0, 0

    def next_train(self, batch_size=1, fix=False):
        start, end = self.idx_train, self.idx_train+batch_size
        x_train, y_train = self.x_train[start:end], self.y_train[start:end]
        x_train = np.expand_dims(x_train, axis=3)

        terminator = False
        if (end >= self.num_train):
            terminator = True
            self.idx_train = 0
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        else:
            self.idx_train = end

        if (fix):
            self.idx_train = start

        if (x_train.shape[0] != batch_size):
            x_train, y_train = self.x_train[-1-batch_size:-1], self.y_train[-1-batch_size:-1]
            x_train = np.expand_dims(x_train, axis=3)

        if (self.normalize):
            min_x, max_x = x_train.min(), x_train.max()
            x_train = (x_train - min_x) / (max_x - min_x)

        x_train_torch = torch.from_numpy(np.transpose(x_train, (0, 3, 1, 2)))
        y_train_torch = torch.from_numpy(y_train)

        return x_train, x_train_torch, y_train, y_train_torch, terminator

    def next_test(self, batch_size=1):
        start, end = self.idx_test, self.idx_test+batch_size
        x_test, y_test = self.x_test[start:end], self.y_test[start:end]
        x_test = np.expand_dims(x_test, axis=3)

        terminator = False
        if (end >= self.num_test):
            terminator = True
            self.idx_test = 0
        else:
            self.idex_test = end

        if (self.normalize):
            min_x, max_x = x_test.min(), x_test.max()
            x_test = (x_test - min_x) / (max_x - min_x)

        x_test_torch = torch.from_numpy(np.transpose(x_test, (0, 3, 1, 2)))
        y_test_torch = torch.from_numpy(y_test)

        return x_test, y_test_torch, y_test, y_test_torch, terminator

