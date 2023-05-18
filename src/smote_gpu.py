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
from imblearn.over_sampling import SMOTE
import numpy as np


def oversample(sampling_strategy, x_train, x_val, y_train, y_val, random_state):
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    x_train = cd.from_pandas(x_train)
    x_val = cd.from_pandas(x_val)
    y_train = cd.from_pandas(y_train)
    y_val = cd.from_pandas(y_val)

    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)

    f1 = f1_score(y_val.to_numpy(), y_pred.to_numpy())
    print(f1)
    print(confusion_matrix(y_val.to_numpy(), y_pred.to_numpy()))

    return f1


def get_f1(sampling_strategy, x_train, x_val, y_train, y_val):
    results = []
    results.append(oversample(sampling_strategy, x_train, x_val, y_train, y_val, 42))
    results.append(oversample(sampling_strategy, x_train, x_val, y_train, y_val, 1))
    results.append(oversample(sampling_strategy, x_train, x_val, y_train, y_val, 18))

    mean_f1 = sum(results) / len(results)
    print("Média F1: ", mean_f1)

    return mean_f1


"""
with open('params/features.pkl', 'rb') as inp:
    features = pickle.load(inp)

search_space = hp.uniform('i', 1, 300)
"""


def objective_oversample(params):
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(nested=True):
        mlflow.autolog()
        sampling_strategy = params
        print("Sampling strategy:", sampling_strategy)
        x_train, y_train, x_val, y_val = data

        # x_train, x_val = preprocessing('all', x_train, x_val)

        f1 = get_f1(sampling_strategy, x_train, x_val, y_train, y_val)

        mlflow.log_param("stage", "Smote_strategy")
        mlflow.log_param("model", "knn")
        mlflow.log_param("sampling_strategy", sampling_strategy)
        mlflow.log_param("model_selection", "train/test-80/20")
        mlflow.log_param("shape", x_train.shape)
        mlflow.log_metric("f1_val", f1)

        # Because fmin() tries to minimize the objective,
        # this function must return the negative accuracy.
        return {"loss": -f1, "status": STATUS_OK}


def update_feature(features):
    with open("params/features.pkl", "wb") as outp:
        pickle.dump(features, outp, pickle.HIGHEST_PROTOCOL)


def update_smote(over_proportion):
    with open("params/smote.pkl", "wb") as outp:
        pickle.dump(over_proportion, outp, pickle.HIGHEST_PROTOCOL)


def get_best_feature():
    with open("params/features.pkl", "rb") as inp:
        features = pickle.load(inp)

    return features


def find_best_sampling_strategy(df, evals):
    train, val = train_test_split(df, test_size=0.2, shuffle=False)
    del df
    train_x = train.drop(["INDISPONIBILIDADE"], axis=1)
    val_x = val.drop(["INDISPONIBILIDADE"], axis=1)
    train_y = train[["INDISPONIBILIDADE"]]
    val_y = val[["INDISPONIBILIDADE"]]

    train_x, val_x = preprocessing("filter", train_x, val_x)

    global data
    data = [train_x, train_y, val_x, val_y]
    rstate = np.random.default_rng(42)
    trials = Trials()
    best_result = fmin(
        fn=objective_oversample,
        space=hp.uniform("sampling_strategy", 0.0045, 0.5),
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials,
        rstate=rstate,
    )

    sampling_strategy = hyperopt.space_eval(
        hp.uniform("sampling_strategy", 0.0038, 0.5), best_result
    )
    print("Best in Search Space:", sampling_strategy)
    print("trials:")
    for trial in trials.trials[:2]:
        print(trial)

    return sampling_strategy


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

    x_train, x_test = preprocessing("filter", x_train, x_test)

    sampling_strategy = find_best_sampling_strategy(x_train, y_train)
    print(sampling_strategy)
