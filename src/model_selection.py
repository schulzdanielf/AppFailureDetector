from sklearn.model_selection import KFold, TimeSeriesSplit
import cupy as cd
from src.preprocessing import preprocessing
from src.eval import eval_metrics
from src.data.data_funcs import split, train_test


def model_selection(m, x, y, p, c, clf):
    if m == "train_test":
        # Ratio train test split
        r = 0.8
        return train_test_selection(r, x, y, p, c, clf)
    elif m == "kfold":
        # K Number of folds
        k = 3
        return kfold_selection(k, x, y, p, c, clf)
    elif m == "time_series":
        # K Number of splits
        k = 3
        return time_series_selection(k, x, y, p, c, clf)


def train_test_selection(r, x, y, p, c, clf):
    df = cd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    train, test = split(r, df)
    x_train, y_train, x_val, y_val = train_test(train, test)

    x_train, x_val, y_train = preprocessing(p, x_train, x_val, y_train)

    if c in ["nb", "dt", "ada"]:
        x_train, x_val = x_train.to_pandas(), x_val.to_pandas()
        y_train, y_val = y_train.to_pandas(), y_val.to_pandas()
    """
    if c == 'lstm':
        return test_model(clf, x_train.copy(), y_train.copy(),
                          x_val.copy(), y_val)"""

    y_train = y_train["INDISPONIBILIDADE"].values

    clf.fit(x_train, y_train)
    del x_train, y_train

    predicted = clf.predict(x_val)
    del x_val

    if c in ["svm", "knn", "rf"]:
        y_val, predicted = y_val.to_numpy(), predicted.to_numpy()

    f1 = eval_metrics(y_val, predicted)[0]
    del y_val, predicted

    return f1


def kfold_selection(k, x, y, p, c, clf):
    results = []
    kf = KFold(n_splits=k)
    for i, (train_index, val_index) in enumerate(kf.split(x)):
        print("Fold:" + str(i))
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        x_train, x_val, y_train = preprocessing(p, x_train, x_val, y_train)

        if c in ["nb", "dt", "ada"]:
            x_train, x_val = x_train.to_pandas(), x_val.to_pandas()
            y_train, y_val = y_train.to_pandas(), y_val.to_pandas()
        y_train = y_train["INDISPONIBILIDADE"].values

        clf.fit(x_train, y_train)
        del x_train, y_train
        predicted = clf.predict(x_val)
        del x_val

        if c in ["svm", "knn", "rf"]:
            y_val, predicted = y_val.to_numpy(), predicted.to_numpy()
        f1 = eval_metrics(y_val, predicted)[0]
        del y_val, predicted
        results.append(f1)

    f1_mean = sum(results) / len(results)

    return f1_mean


def time_series_selection(k, x, y, p, c, clf):
    results = []
    kf = TimeSeriesSplit(n_splits=k, test_size=115000)
    for i, (train_index, val_index) in enumerate(kf.split(x)):
        print("Fold:" + str(i))
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        x_train, x_val, y_train = preprocessing(p, x_train, x_val, y_train)

        if c in ["nb", "dt", "ada"]:
            x_train, x_val = x_train.to_pandas(), x_val.to_pandas()
            y_train, y_val = y_train.to_pandas(), y_val.to_pandas()
        y_train = y_train["INDISPONIBILIDADE"].values

        clf.fit(x_train, y_train)
        del x_train, y_train
        predicted = clf.predict(x_val)
        del x_val

        if c in ["svm", "knn", "rf"]:
            y_val, predicted = y_val.to_numpy(), predicted.to_numpy()
        f1 = eval_metrics(y_val, predicted)[0]
        del y_val, predicted
        results.append(f1)

    f1_mean = sum(results) / len(results)

    return f1_mean
