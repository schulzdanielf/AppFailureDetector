from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import cudf as cd
import cupy as cp
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)
import numpy as np


def get_features(x, y, i):
    anova_selector = SelectKBest(f_classif, k=i)
    anova_selector.fit(x, y)
    anova_support = anova_selector.get_support()
    anova_feature = x.loc[:, anova_support].columns.tolist()

    return anova_feature


def best_features(train_x, val_x, train_y, val_y, i):
    print("I: ", i)
    results = []
    anova_selector = SelectKBest(f_classif, k=i - 1)
    anova_selector.fit(train_x, train_y)
    anova_support = anova_selector.get_support()
    anova_feature = train_x.loc[:, anova_support].columns.tolist()

    f1 = get_f1(train_x[anova_feature], val_x[anova_feature], train_y, val_y)
    results.append(f1)

    anova_selector = SelectKBest(f_classif, k=i + 1)
    anova_selector.fit(train_x, train_y)
    anova_support = anova_selector.get_support()
    anova_feature = train_x.loc[:, anova_support].columns.tolist()

    f1 = get_f1(train_x[anova_feature], val_x[anova_feature], train_y, val_y)
    results.append(f1)

    anova_selector = SelectKBest(f_classif, k=i)
    anova_selector.fit(train_x, train_y)
    anova_support = anova_selector.get_support()
    anova_feature = train_x.loc[:, anova_support].columns.tolist()

    f1 = get_f1(train_x[anova_feature], val_x[anova_feature], train_y, val_y)
    results.append(f1)

    f1 = sum(results) / len(results)

    return f1, anova_feature


def get_f1(x_train, x_val, y_train, y_val):
    x_train = cd.from_pandas(x_train)
    x_val = cd.from_pandas(x_val)
    y_train = cd.from_pandas(y_train)
    y_val = cd.from_pandas(y_val)

    svc = SVC()
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_val)

    f1 = f1_score(y_val.to_numpy(), y_pred.to_numpy())
    print(f1)
    print(confusion_matrix(y_val.to_numpy(), y_pred.to_numpy()))

    return f1


"""
with open('params/features.pkl', 'rb') as inp:
    features = pickle.load(inp)

search_space = hp.uniform('i', 1, 300)
"""


def objective_features(params):
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(nested=True):
        mlflow.autolog()
        i = params
        train_x, train_y, val_x, val_y = data
        f1 = best_features(train_x, val_x, train_y, val_y, i)[0]

        mlflow.log_param("stage", "Feature_extraction")
        mlflow.log_param("model", "svm")
        mlflow.log_param("model_selection", "train_test")
        mlflow.log_param("features_i", i)
        mlflow.log_metric("f1_val", f1)

        # Because fmin() tries to minimize the objective,
        # this function must return the negative accuracy.
        return {"loss": -f1, "status": STATUS_OK}


def update_feature(features):
    with open("params/features.pkl", "wb") as outp:
        pickle.dump(features, outp, pickle.HIGHEST_PROTOCOL)


def get_best_feature():
    with open("params/features.pkl", "rb") as inp:
        features = pickle.load(inp)

    return features


def find_best_features(df, max_evals):
    train, val = train_test_split(df, test_size=0.2, shuffle=False)
    del df
    train_x = train.drop(["INDISPONIBILIDADE"], axis=1)
    val_x = val.drop(["INDISPONIBILIDADE"], axis=1)
    train_y = train[["INDISPONIBILIDADE"]]
    val_y = val[["INDISPONIBILIDADE"]]

    global data
    data = [train_x, train_y, val_x, val_y]
    rstate = np.random.default_rng(42)
    trials = Trials()
    best_result = fmin(
        fn=objective_features,
        space=hp.choice("i", range(30, 150)),
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=rstate,
    )

    i = hyperopt.space_eval(hp.choice("i", range(30, 150)), best_result)
    print("Best in Search Space:", i)
    print("trials:")
    for trial in trials.trials[:2]:
        print(trial)

    features = best_features(train_x, val_x, train_y, val_y, i)[1]

    return features


def split(mat):
    i = int(len(mat) * 0.8)

    return mat[:i], mat[i:]


def preprocess(filtering, scaler, train_x, test_x):
    if filtering == "True":
        print("Filtering")
        with open("params/features.pkl", "rb") as inp:
            features = pickle.load(inp)
        train_x = train_x[features]
        test_x = test_x[features]

    if scaler == "True":
        print("Standard Scale")
        ss = StandardScaler()
        ss.fit(train_x)
        train_x = ss.transform(train_x)
        test_x = ss.transform(test_x)

    return train_x, test_x


def preprocessing(p, train_x, test_x):
    if p == "all":
        train_x, test_x = preprocess("True", "True", train_x, test_x)
    elif p == "filter":
        train_x, test_x = preprocess("True", "False", train_x, test_x)
    elif p == "scaler":
        train_x, test_x = preprocess("False", "True", train_x, test_x)

    return train_x, test_x


def get_data():
    mat = pd.read_csv("data/matomo.csv", dtype=cp.float32)

    return mat


def train_test(train, test):
    x_train = train.drop(["INDISPONIBILIDADE"], axis=1)
    y_train = train[["INDISPONIBILIDADE"]]

    x_test = test.drop(["INDISPONIBILIDADE"], axis=1)
    y_test = test[["INDISPONIBILIDADE"]]

    return x_train, y_train, x_test, y_test


def eval_metrics(actual, pred):
    f1 = f1_score(actual, pred)
    roc = roc_auc_score(actual, pred)
    rec = recall_score(actual, pred)
    pre = precision_score(actual, pred)
    acc = accuracy_score(actual, pred)
    print("F1-Score:", f1)
    print(confusion_matrix(actual, pred))
    return f1, roc, rec, pre, acc


def run():
    mat = get_data()

    train, test = split(mat)

    x_train, y_train, x_test, y_test = train_test(train, test)

    features = find_best_features(x_train, y_train)
    print(features)
