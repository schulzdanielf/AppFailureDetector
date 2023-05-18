import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import cudf as cd
import cupy as cp
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.naive_bayes import GaussianNB
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
import json
import mlflow
import pickle
from imblearn.over_sampling import SMOTE


search_space = hp.choice(
    "classifier_type",
    [
        {
            "type": "rf",
            "n_estimators": hp.quniform("n_estimators", 1, 1000, 1),
            "preprocessing": hp.choice(
                "p_rf",
                ["scaler", "filter", "all", "none", "fi_ss", "fi_sm", "ss_sm", "smote"],
            ),
        },
        {
            "type": "knn",
            "n_neighbors": hp.choice("n_neighbors", [1, 3, 5, 7, 9]),
            "metric": hp.choice("metric", ["euclidean", "manhattan", "minkowski"]),
            "preprocessing": hp.choice(
                "p_knn",
                ["scaler", "filter", "all", "none", "fi_ss", "fi_sm", "ss_sm", "smote"],
            ),
        },
        {
            "type": "svm",
            "C": hp.uniform("C", 0.01, 100),
            "kernel": hp.choice("kernel", ["rbf", "sigmoid"]),
            "preprocessing": hp.choice(
                "p_svm",
                ["scaler", "filter", "all", "none", "fi_ss", "fi_sm", "ss_sm", "smote"],
            ),
        },
    ],
)


"""
    {
        'type': 'knn',
        'n_neighbors': hp.choice('n_neighbors', [1, 3, 5, 7, 9]),
        'metric': hp.choice('metric', ['euclidean', 'manhattan', 'minkowski']),
        'preprocessing': hp.choice('p_knn',
                                   ['scaler', 'filter', 'all', 'none', 'fi_ss',
                                   'fi_sm', 'ss_sm', 'smote'])
    },
    {
        'type': 'svm',
        'C': hp.choice('C', [0.01, 0.1, 0.5, 1, 10, 100]),
        'kernel': hp.choice('kernel', ['rbf', 'sigmoid']),
        'preprocessing': hp.choice('p_svm',
                                   ['scaler', 'filter', 'all', 'none', 'fi_ss',
                                   'fi_sm', 'ss_sm', 'smote'])
    },
    {
        'type': 'rf',
        'n_estimators': hp.choice('n_estimators', [25, 50, 100, 200, 500]),
        'preprocessing': hp.choice('p_rf',
                                   ['scaler', 'filter', 'all', 'none'])
    },
    {
        'type': 'nb',
        'var_smoothing': hp.choice('var_smoothing', [1e-9, 1e-5, 1e-20]),
        'preprocessing': hp.choice('p_nb',
                                   ['scaler', 'filter', 'all', 'none'])
    },
"""


def eval_metrics(actual, pred):
    f1 = f1_score(actual, pred)
    roc = roc_auc_score(actual, pred)
    rec = recall_score(actual, pred)
    pre = precision_score(actual, pred)
    acc = accuracy_score(actual, pred)
    print("F1-Score:", f1)
    print(confusion_matrix(actual, pred))
    return f1, roc, rec, pre, acc


def objective(params):
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(nested=True):
        mlflow.autolog()
        mlflow.log_params(params)
        mlflow.log_param("model_selection", "train/test-80/20")

        x_train, y_train, x_val, y_val = data
        p = params["preprocessing"]
        print("Preprocess:", p)
        x_train, x_val, y_train = preprocessing(p, x_train, x_val, y_train)

        classifier_type = params["type"]

        mlflow.log_param("model", classifier_type)
        del params["type"]
        del params["preprocessing"]
        if classifier_type == "knn":
            clf = KNeighborsClassifier(**params)
        elif classifier_type == "svm":
            clf = SVC(**params)
        elif classifier_type == "nb":
            clf = GaussianNB(**params)
        elif classifier_type == "rf":
            clf = RandomForestClassifier(**params)
        else:
            return 0
        print(params)

        print("Initiating fiting the model:", classifier_type)
        y_train = y_train["INDISPONIBILIDADE"].values

        clf.fit(x_train, y_train)
        print("Finish training")
        predicted = clf.predict(x_val)
        f1, roc, rec, pre, acc = eval_metrics(y_val.to_numpy(), predicted.to_numpy())

        mlflow.log_metric("f1_val", f1)
        mlflow.log_metric("roc_val", roc)
        mlflow.log_metric("recall_val", rec)
        mlflow.log_metric("precision_val", pre)
        mlflow.log_metric("accuracy_val", acc)

        # Because fmin() tries to minimize the objective,
        # this function must return the negative accuracy.
        return {"loss": -f1, "status": STATUS_OK}


def get_best(key):
    f = open("params/best_hyper.json")
    data = json.load(f)
    f.close()
    return data[key]


def find_best(x, y):
    df = cd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    train, val = split(df)

    del df

    x_train = train.drop(["INDISPONIBILIDADE"], axis=1)
    x_val = val.drop(["INDISPONIBILIDADE"], axis=1)
    y_train = train[["INDISPONIBILIDADE"]]
    y_val = val[["INDISPONIBILIDADE"]]

    global data
    data = [x_train, y_train, x_val, y_val]

    trials = Trials()
    best_result = fmin(
        fn=objective, space=search_space, algo=tpe.suggest, max_evals=200, trials=trials
    )

    result = hyperopt.space_eval(search_space, best_result)
    print("Best in Search Space:", result)
    print("trials:")
    for trial in trials.trials[:2]:
        print(trial)

    key = result["type"]
    del result["type"]
    # update_hyper(result, key)

    best = get_best(key)
    p = result["preprocessing"]
    print("Model:" + key + "Preprocessing" + p)
    print(result)

    x_train, x_val, y_train = preprocessing(p, x_train, x_val, y_train)
    y_train = y_train["INDISPONIBILIDADE"].values
    #    del best['preprocessing']
    print("Get best", best)
    if key == "knn":
        model = KNeighborsClassifier(**best)
    elif key == "svm":
        model = SVC(**best)
    elif key == "rf":
        model = RandomForestClassifier(**best)
    elif key == "nb":
        model = GaussianNB(**best)

    print("Fiting model")
    model.fit(x_train, y_train)
    return model, result, trials, key


def split(mat):
    i = int(len(mat) * 0.8)

    return mat[:i], mat[i:]


def preprocess(filtering, scaler, smote, x_train, x_test, y_train):
    if filtering == "True":
        print("Filtering")
        with open("params/features.pkl", "rb") as inp:
            features = pickle.load(inp)
        x_train = x_train[features]
        x_test = x_test[features]

    if scaler == "True":
        print("Standard Scale")
        ss = StandardScaler()
        ss.fit(x_train)
        x_train = ss.transform(x_train)
        x_test = ss.transform(x_test)

    if smote == "True":
        print("SMOTE")
        x_train, y_train = x_train.to_pandas(), y_train.to_pandas()

        smote = SMOTE(random_state=42, sampling_strategy=0.01715547654473271)
        x_train, y_train = smote.fit_resample(x_train, y_train)

        x_train, y_train = cd.from_pandas(x_train), cd.from_pandas(y_train)

    return x_train, x_test, y_train


def preprocessing(p, x_train, x_test, y_train):
    if p == "all":
        x_train, x_test, y_train = preprocess(
            "True", "True", "True", x_train, x_test, y_train
        )
    elif p == "filter":
        x_train, x_test, y_train = preprocess(
            "True", "False", "False", x_train, x_test, y_train
        )
    elif p == "scaler":
        x_train, x_test, y_train = preprocess(
            "False", "True", "False", x_train, x_test, y_train
        )
    elif p == "smote":
        x_train, x_test, y_train = preprocess(
            "False", "False", "True", x_train, x_test, y_train
        )
    elif p == "fi_sm":
        x_train, x_test, y_train = preprocess(
            "True", "False", "True", x_train, x_test, y_train
        )
    elif p == "fi_ss":
        x_train, x_test, y_train = preprocess(
            "True", "True", "False", x_train, x_test, y_train
        )
    elif p == "ss_sm":
        x_train, x_test, y_train = preprocess(
            "False", "True", "True", x_train, x_test, y_train
        )

    return x_train, x_test, y_train


def get_data():
    mat = cd.read_csv("data/matomo.csv", dtype=cp.float32)

    return mat


def train_test(train, test):
    x_train = train.drop(["INDISPONIBILIDADE"], axis=1)
    y_train = train[["INDISPONIBILIDADE"]]

    x_test = test.drop(["INDISPONIBILIDADE"], axis=1)
    y_test = test[["INDISPONIBILIDADE"]]

    return x_train, y_train, x_test, y_test


def run():
    print("Reading data")
    mat = get_data()
    train, test = split(mat)
    x_train, y_train, x_test, y_test = train_test(train, test)

    print("Finding best model")
    model, result, trials, k = find_best(x_train, y_train)
    print(result)
    p = result["preprocessing"]
    classifier_type = k

    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(nested=True):
        mlflow.log_param("preprocessing", p)
        mlflow.log_param("model", classifier_type)
        x_train, x_test, y_train = preprocessing(p, x_train, x_test, y_train)

        pred = model.predict(x_test)

        f1, roc, rec, pre, acc = eval_metrics(y_test.to_numpy(), pred.to_numpy())

        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc", roc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", pre)
        mlflow.log_metric("accuracy", acc)


run()
