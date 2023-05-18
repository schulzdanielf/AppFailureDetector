import mlflow
from src.preprocessing import preprocessing
from src.eval import eval_metrics
import numpy as np
import statsmodels.stats.api as sms
from src.models.classifiers import get_model
import cudf as cd


def is_sklearn(model):
    if type(model).__module__[:7] == "sklearn":
        return True
    return False


def eval_one_variance(test, model, key):
    results = []
    for i in range(10):
        with mlflow.start_run(nested=True):
            mlflow.log_param("model", key)
            mlflow.log_param("stage", "statistics_analysis")
            # mlflow.log_param("model_selection", split_strategy)
            mlflow.log_param("random_i", i)

            test_shuffle = test.sample(frac=0.5, random_state=i)

            x_test = test_shuffle.drop(["INDISPONIBILIDADE"], axis=1)
            y_test = test_shuffle[["INDISPONIBILIDADE"]].to_pandas()
            # print("Y_TEST:",y_test)
            if is_sklearn(model):
                x_test = x_test.to_pandas()

            pred = model.predict(x_test)

            if not is_sklearn(model):
                pred = pred.to_numpy()
                y_test = y_test["INDISPONIBILIDADE"].values

            f1, roc, rec, pre, acc = eval_metrics(y_test, pred)
            results.append(f1)

            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc", roc)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("precision", pre)
            mlflow.log_metric("accuracy", acc)

    media = np.mean(results)
    dp = np.std(results, ddof=1)
    ci = sms.DescrStatsW(results).tconfint_mean()
    return media, dp, ci


def eval_variance(x_train, y_train, x_test, y_test, params):
    metricas = {}
    for k in params:
        if k not in ["lstm"]:
            print("Algo:", k)
            p = params[k]["preprocessing"]
            del params[k]["preprocessing"]
            (
                x_train_c,
                x_test_c,
                y_train_c,
            ) = preprocessing(p, x_train, x_test, y_train)

            model = get_model(k, params[k])
            if is_sklearn(model):
                x_train_c, y_train_c = x_train_c.to_pandas(), y_train_c.to_pandas()

            # if k == "lstm":
            #    print("Transforming data to LSTM")
            #    x_train_c, y_train_c = x_train_c.to_pandas(), y_train_c.to_pandas()
            #    print("Ajusting y")
            #    y_train_c = ajusta_y(y_train)
            #    print("Transforming dimension")
            #    x_train_c, y_train_c = transform_dimension_timesteps(x_train, y_train, time_steps=1)

            y_train_c = y_train_c["INDISPONIBILIDADE"].values
            print("Fitting model")
            model.fit(x_train_c, y_train_c)

            test = cd.concat(
                [x_test_c.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
            )

            media, dp, ci = eval_one_variance(test, model, k)
            metricas[k] = {}
            metricas[k]["mean"] = media
            metricas[k]["stand_dev"] = dp
            metricas[k]["conf_int"] = ci
    return metricas
