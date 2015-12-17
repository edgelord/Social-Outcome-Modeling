from itertools import combinations_with_replacement
from header import schema
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import pydot
from sklearn.metrics import mean_squared_error, mean_absolute_error


from sklearn.externals.six import StringIO
from sklearn.linear_model import LinearRegression, Lasso, ElasticNetCV, LassoCV, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


def load_data(filename='../rsrc/data.csv', numeric=True):
    df = pd.read_csv(filename, names=schema)
    idx = df.communityname
    if numeric:
        return df._get_numeric_data()
    else:
        return df


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


def show_linear_results(feats, model):
    imp = important_features(feats, model.coef_)
    print 'Associated features: '
    show(imp[:20])
    print 'Disassociated features: '
    show(imp[-20:][::-1])


def train_model(feats, x, y, model, split_ratio=.8):
    split = len(y) * split_ratio
    model.fit(x[:split], y[:split])

    err = model.predict(x[split:]) - y[split:]
    sq_err = err ** 2
    print 'Average Error: ', np.average(np.abs(err))
    print 'Avg. Rel. Error: ', np.average(np.abs(err)) / np.average(y[split:])
    if type(model) in [Lasso, LinearRegression]:
        show_linear_results(feats, model)
    return model


def lin_reg(feats, x, y):
    m = LinearRegression(n_jobs=-1, normalize=True)
    return train_model(feats, x, y, m)


def lasso(feats, x, y, alpha=.0005, iters=3000):
    m = Lasso(alpha=alpha, max_iter=iters, normalize=True)
    # m = LassoCV(eps=.00001, n_jobs=-1)
    return train_model(feats, x, y, m)


def enet(feats, x, y, alpha=.0005, iters=3000):
    m = ElasticNetCV()
    return train_model(feats, x, y, m)


def regtree(feats, x, y):
    m = DecisionTreeRegressor(max_depth=3)
    m = train_model(feats, x, y, m)
    return m


def rf(feats, x, y):
    m = RandomForestRegressor()
    m = train_model(feats, x, y, m)
    return m


def svm(feats, x, y):
    m = SVR()
    m = train_model(feats, x, y, m)
    return m


def knn(feats, x, y):
    m = KNeighborsRegressor(n_neighbors=10, weights='distance', leaf_size=20)
    m = train_model(feats, x, y, m)
    return m


def experiment(df, dv, model=lasso, feats='quadratic'):
    if feats == 'linear':
        IVs = df.drop(dv, 1).values
        feats = df.drop(dv, 1).columns.values
    else:
        feats, IVs = quadratic_features(df.drop(dv, 1))
    y = df[dv].values
    return model(feats, IVs, y)


def show_tree(clf, feature_names, file_name):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_data.getvalue(),)
    graph.write_pdf(file_name)

'''('MalePctDivorce*PctPersDenseHous', 0.30729275274456691)
('racepctblack*PctHousLess3BR', 0.24991535001964477)
('HispPerCap*NumStreet', 0.14249808478068951)
('PctHousNoPhone*MedRent', 0.094030129820233843)
('pctWPubAsst*whitePerCap', 0.086722631950222673)
('pctUrban*PctIlleg', 0.06935651216941284)
('PctIlleg*PctHousLess3BR', 0.063355648990797805)
('MalePctDivorce*PctIlleg', 0.056406095389051503)
('PctHousLess3BR*HousVacant', 0.050754318447706268)
('PctIlleg*PctSameCity85', 0.044622293970025581)
('MalePctDivorce*PctVacantBoarded', 0.041138115481024327)
('MalePctDivorce*HousVacant', 0.035023958900577037)
('FemalePctDiv*MedRentPctHousInc', 0.032977637133629764)
('pctWPubAsst*MalePctDivorce', 0.019802316010702738)
('FemalePctDiv*PctIlleg', 0.019368969618522035)
('TotalPctDiv*MedRentPctHousInc', 0.015073230757971201)
('whitePerCap*PctPopUnderPov', 0.0099085617576258628)
'''


def binarize(Y, percentile):
    thresh = np.percentile(Y, percentile)
    if percentile < 50:
        return Y < thresh
    return Y > thresh


def rmse(pred, truth):
    np.abs(pred) - np.truth

from collections import defaultdict
class NNLR:

    def __init__(self, k=5, rad=2, mode='k', feat_names=None):
        self.mode = mode
        self.k = k
        self.NN = NearestNeighbors(k, radius=rad)
        
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.NN.fit(X)
        self.active=defaultdict(int)
    def nn_lin(self, testX, neighbors):
        l = DecisionTreeRegressor()
        # return np.mean(self.Y[neighbors])
        l.fit(self.X[neighbors], self.Y[neighbors])
        # for idx in np.where(l.coef_)[0]:
            # self.active[idx]+=1
        return l.predict([testX])[0]

    def predict(self, X):
        if self.mode == 'k':
            neighbors = self.NN.kneighbors(X)[1]
        elif self.mode == 'rad':
            neighbors = self.NN.radius_neighbors(X)[1]
        return np.array([self.nn_lin(Xtst, nbr) for (Xtst, nbr) in zip(X, neighbors)])


def tst(X, Y, k=3, rad=4, mode='k'):
    trX = X[:-1200]
    trY = Y[:-1200]
    tstX = X[-400:]
    tstY = Y[-400:]

    nnlr = NNLR(k, rad, mode)

    nnlr.fit(trX, trY)

    pred = nnlr.predict(trX)
    print 'Training Set'
    print 'Root Mean Squared Error'
    print mean_squared_error(trY, pred)**.5
    print 'Root Mean Error'
    print mean_absolute_error(trY, pred)
    # print zip(pred, trX)[:5]
    print nnlr.active

    pred = nnlr.predict(tstX)
    print 'Test Set'
    print 'Root Mean Squared Error'
    print mean_squared_error(tstY, pred)**.5
    print 'Root Mean Error'
    print mean_absolute_error(tstY, pred)
    # print zip(pred, tstY)[:5]
    print nnlr.active
