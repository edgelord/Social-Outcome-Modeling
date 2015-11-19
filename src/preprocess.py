from itertools import combinations_with_replacement
from header import schema

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures


def load_data(filename='../rsrc/data.csv'):
    df = pd.read_csv(filename, names=schema)
    idx = df.communityname
    return df._get_numeric_data()


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


def quadratic_features(df):
    df = df.copy()
    cols = df.columns.values
    for a, b in combinations_with_replacement(cols, 2):
        feat = a + '*' + b
        df[feat] = df[a] * df[b]
    return df.columns.values, df.values


def important_features(features, coeffs):
    return sorted(zip(features, coeffs), key=lambda x: x[1], reverse=True)


def show(vec):
    for x in vec:
        print x


def train_model(feats, x, y, model, split_ratio=.8):
    split = len(y) * split_ratio
    model.fit(x[:split], y[:split])

    err = model.predict(x[split:]) - y[split:]
    sq_err = err ** 2
    print 'Average Error: ', np.average(np.abs(err))
    print 'Avg. Rel. Error: ', np.average(err/y[split:])
    imp = important_features(feats, model.coef_)
    print 'Associated features: '
    show(imp[:10])
    print 'Disassociated features: '
    show(imp[-10:][::-1])
    return model


def lin_reg(feats, x, y):
    m = LinearRegression(n_jobs=-1, normalize=True)
    m = train_model(feats, x, y, m)


def lasso(feats, x, y, alpha=.0001, iters=3000):
    m = Lasso(alpha=alpha, max_iter=iters, normalize=True)
    m = train_model(feats, x, y, m)


def experiment(df, iv):
    feats, qx = quadratic_features(df.drop(iv,1))
    y = df[iv].values
    lasso(feats,qx,y)
