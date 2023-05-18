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

# from sklearn.model_selection import KFold
from sklearn.model_selection import KFold, TimeSeriesSplit


def best_features(x, y, i):
    print("I: ", i)
    results = []
    anova_selector = SelectKBest(f_classif, k=i - 1)
    anova_selector.fit(x, y)
    anova_support = anova_selector.get_support()
    anova_feature = x.loc[:, anova_support].columns.tolist()

    f1 = get_f1(x[anova_feature], y)
    results.append(f1)

    anova_selector = SelectKBest(f_classif, k=i + 1)
    anova_selector.fit(x, y)
    anova_support = anova_selector.get_support()
    anova_feature = x.loc[:, anova_support].columns.tolist()

    f1 = get_f1(x[anova_feature], y)
    results.append(f1)

    anova_selector = SelectKBest(f_classif, k=i)
    anova_selector.fit(x, y)
    anova_support = anova_selector.get_support()
    anova_feature = x.loc[:, anova_support].columns.tolist()

    f1 = get_f1(x[anova_feature], y)
    results.append(f1)

    f1 = sum(results) / len(results)

    return f1, anova_feature


def get_f1(x, y):
    results = []

    if model_selection == "kfold":
        kf = KFold(n_splits=3)
    elif model_selection == "time_series":
        kf = TimeSeriesSplit(n_splits=3, test_size=115000)

    for i, (train_index, val_index) in enumerate(kf.split(x)):
        print("Fold:" + str(i))

        x_train = cd.from_pandas(x.iloc[train_index])
        x_val = cd.from_pandas(x.iloc[val_index])
        y_train = cd.from_pandas(y.iloc[train_index])
        y_val = cd.from_pandas(y.iloc[val_index])

        svc = SVC()
        svc.fit(x_train, y_train)
        del x_train, y_train
        y_pred = svc.predict(x_val)
        del x_val
        f1 = f1_score(y_val.to_numpy(), y_pred.to_numpy())
        print(f1)
        print(confusion_matrix(y_val.to_numpy(), y_pred.to_numpy()))
        del y_val, y_pred
        results.append(f1)
    f1_mean = sum(results) / len(results)
    return f1_mean


"""
with open('params/features.pkl', 'rb') as inp:
    features = pickle.load(inp)

search_space = hp.uniform('i', 1, 300)
"""


def objective(params):
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(nested=True):
        mlflow.autolog()
        i = params
        x, y = data
        f1 = best_features(x, y, i)[0]

        mlflow.log_param("model", "svm")
        mlflow.log_param("features_i", i)
        mlflow.log_param("model_selection", model_selection)
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


def find_best_cross_features(df, evals, ms):
    global model_selection
    model_selection = ms

    train, val = train_test_split(df, test_size=0.2, shuffle=False)
    del df
    train_x = train.drop(["INDISPONIBILIDADE"], axis=1)
    # val_x = val.drop(["INDISPONIBILIDADE"], axis=1)
    train_y = train[["INDISPONIBILIDADE"]]
    # val_y = val[["INDISPONIBILIDADE"]]

    global data
    data = [train_x, train_y]

    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=hp.choice("i", range(30, 120)),
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials,
    )

    i = hyperopt.space_eval(hp.choice("i", range(30, 120)), best_result)
    print("Best in Search Space:", i)
    print("trials:")
    for trial in trials.trials[:2]:
        print(trial)

    features = best_features(train_x, train_y, i)[1]
    # update_feature(features)
    # best = get_best_feature()
    # print(best)

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

    features = find_best_cross_features(x_train, y_train)
    print(features)
