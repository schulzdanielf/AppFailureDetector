from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import pickle
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split


def best_features(train_x, val_x, train_y, val_y, i):
    anova_selector = SelectKBest(f_classif, k=i)
    anova_selector.fit(train_x, train_y)
    anova_support = anova_selector.get_support()
    anova_feature = train_x.loc[:, anova_support].columns.tolist()

    f1 = get_f1(train_x[anova_feature], val_x[anova_feature], train_y, val_y)

    return f1, anova_feature


def get_f1(X_train, X_val, y_train, y_val):
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    return f1


with open("/app/params/features.pkl", "rb") as inp:
    features = pickle.load(inp)

search_space = hp.uniform("i", 1, 300)


def objective(params):
    with mlflow.start_run(nested=True):
        mlflow.autolog()
        i = params
        train_x, train_y, val_x, val_y = data
        f1 = best_features(train_x, val_x, train_y, val_y, i)[0]

        mlflow.log_param("features_i", i)
        mlflow.log_metric("f1", f1)

        # Because fmin() tries to minimize the objective,
        # this function must return the negative accuracy.
        return {"loss": -f1, "status": STATUS_OK}


def update_feature(features):
    with open("/app/params/features.pkl", "wb") as outp:
        pickle.dump(features, outp, pickle.HIGHEST_PROTOCOL)


def get_best_feature():
    with open("/app/params/features.pkl", "rb") as inp:
        features = pickle.load(inp)

    return features


def find_best_features(x, y):
    df = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    del x, y
    train, val = train_test_split(df, test_size=0.2, shuffle=False)
    del df
    train_x = train.drop(["INDISPONIBILIDADE"], axis=1)
    val_x = val.drop(["INDISPONIBILIDADE"], axis=1)
    train_y = train[["INDISPONIBILIDADE"]]
    val_y = val[["INDISPONIBILIDADE"]]

    global data
    data = [train_x, train_y, val_x, val_y]

    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=hp.choice("i", range(20, 40)),
        algo=tpe.suggest,
        max_evals=5,
        trials=trials,
    )

    i = hyperopt.space_eval(hp.choice("i", range(20, 40)), best_result)
    print("Best in Search Space:", i)
    print("trials:")
    for trial in trials.trials[:2]:
        print(trial)

    features = best_features(train_x, val_x, train_y, val_y, i)[1]
    update_feature(features)
    best = get_best_feature()
    print(best)
