from itertools import combinations_with_replacement
from header import schema

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def load_data(filename='../rsrc/data.csv'):
    df = pd.read_csv(filename, names=schema)
    idx = df.communityname
    return df._get_numeric_data()


def force_2d(arr):
    if len(arr.shape) is 1:
        arr = arr.reshape(len(arr), 1)
    return arr


def select_columns(df, y_col, x_cols=None, quadratic=True):
    'Returns a dataset in form x, y'
    x = df.drop(y_col, 1) if x_cols is None else df[x_cols]
    y = df[y_col]
    return np.array(x), np.array(y)


def partition_dataset(dataset, split_prop=.8):
    xs, ys = dataset
    data_size = len(xs)
    split = int(data_size * split_prop)
    train = xs[:split], ys[:split]
    test = xs[split:], ys[split:]
    return train, test

arr = np.arange(0, 100).reshape(20, 5)
tdf = pd.DataFrame(arr, columns=['a', 'b', 'c', 'd', 'e'])


def quadratic_features(df, biased=True):
    df = df.copy()
    cols = df.columns.values
    for a, b in combinations_with_replacement(cols, 2):
        feat = a + '*' + b
        df[feat] = df[a] * df[b]
    df['bias'] = np.ones(len(df))
    return df.columns.values, df.values


def important_features(features, coeffs):
    return sorted(zip(features, coeffs), key=lambda x: x[1])


def experiment(df, y_col, split_ratio=.8):
    x = df.drop(y_col, 1)
    feats, quad_x = quadratic_features(x)
    y = np.array(df[y_col])

    split = len(y) * split_ratio
    lr = LinearRegression(fit_intercept=False, n_jobs=-1)
    lr.fit(quad_x[:split], y[:split])

    err = lr.predict(quad_x[split:]) - y[split:]
    sq_err = err ** 2
    print 'RMSE: ', sq_err

    imp = important_features(feats, lr.coef_)
    print 'most important features: ', imp[:10]
    return quad_x, lr
