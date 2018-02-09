
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath('..'))
from hmmd_scaled import HMM

from baseline import get_data
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import f1_score
from predict_errors1 import predicterrors
from get_data import get_data_error

def accuracy(T, Y):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total


def total_f1_score(T, Y):
    # inputs are lists of lists
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()


# def flatten(l):
#     return [item for sublist in l for item in sublist]


    # print results
    print("train accuracy:", accuracy(Ytrain, Ptrain))
    print("test accuracy:", accuracy(Ytest, Ptest))
    print("train f1:", total_f1_score(Ytrain, Ptrain))
    print("test f1:", total_f1_score(Ytest, Ptest))



if __name__ == '__main__':
    main()
