import hyperopt
from hyperopt import fmin, tpe, STATUS_OK, Trials
import mlflow
import numpy as np
from src.smote_gpu import find_best_sampling_strategy, update_smote
from src.best_results import get_best_parameters
from src.features_gpu import find_best_features
from src.model_selection import model_selection
from src.data.data_funcs import get_data, train_test, split
from src.mlflow_funcs import delete_statistics_analysis
from src.cross_features_gpu import find_best_cross_features, update_feature
from src.search_space import get_search_space
from src.parameters import update_parameters
from src.statistics import eval_variance
from src.models.classifiers import get_model


def objective(params):
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        x, y = data
        p = params["preprocessing"]
        print("Preprocess:", p)
        classifier_type = params["type"]
        print("Model", classifier_type)
        del params["type"]
        del params["preprocessing"]
        mlflow.log_param("model", classifier_type)
        mlflow.log_param("model_selection", split_strategy)
        mlflow.log_param("preprocessing", p)

        clf = get_model(classifier_type, params)
        print(params)

        f1 = model_selection(split_strategy, x, y, p, classifier_type, clf)

        print("Média F1-SCORE", f1)
        mlflow.log_metric("f1_val", f1)

        return {"loss": -f1, "status": STATUS_OK}


def find_best(x, y, evals, space):
    global data
    data = [x, y]
    rstate = np.random.default_rng(42)
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials,
        rstate=rstate,
    )

    result = hyperopt.space_eval(space, best_result)
    print("Best in Search Space:", result)
    print("trials:")
    for trial in trials.trials[:2]:
        print(trial)

    key = result["type"]
    del result["type"]
    # update_hyper(result, key)

    p = result["preprocessing"]
    print("Model:" + key + "Preprocessing" + p)
    print(result)

    return result, trials, key


def run():
    print("Reading data")
    mat = get_data()
    print("Spliting the data into train/test with 75/25 proportion")
    train, test = split(0.75, mat)
    print("Spliting the data into x and y features")
    x_train, y_train, x_test, y_test = train_test(train, test)

    print("Find best features for filtering dataset")

    if split_strategy == "train_test":
        features = find_best_features(train.to_pandas(), 100)
    else:
        features = find_best_cross_features(train.to_pandas(), 100, split_strategy)
    print("Best features finded:", features)
    # features = get_features(x_train.to_pandas(), y_train.to_pandas(), 44)
    update_feature(features)

    print("Find best proportion for oversample dataset")
    over_proportion = find_best_sampling_strategy(train.to_pandas(), 100)
    print("Best proportion finded:", over_proportion)
    update_smote(over_proportion)

    search_space, search_space_cpu, search_space_lstm = get_search_space()

    print("Find best parameters for algorithms in dataset")
    find_best(x_train, y_train, 500, search_space)

    print("Find best parameters for cpu algorithms in dataset")
    find_best(x_train, y_train, 50, search_space_cpu)
    """
    run_keras_optmize()
    """
    # print("Find best parameters for LSTM model")
    # find_best(x_train, y_train, evals, search_space_lstm)

    params = get_best_parameters(split_strategy)
    print("Best parameters finded:", params)
    update_parameters(params)

    # print("Run best parameters on test dataset")
    # test_params(x_train, y_train, x_test, y_test, params)

    params = get_best_parameters(split_strategy)
    print("Eval results for each algo")
    metrics = eval_variance(x_train, y_train, x_test, y_test, params)
    print("Statistics for each algo:", metrics)


delete_statistics_analysis()
print("Reading data")
mat = get_data()
print("Spliting the data into train/test with 75/25 proportion")
train, test = split(0.75, mat)
print("Spliting the data into x and y features")
x_train, y_train, x_test, y_test = train_test(train, test)
split_strategy = "train_test"
# import ipdb; ipdb.set_trace()
params = get_best_parameters(split_strategy)
print(params)
eval_variance(x_train, y_train, x_test, y_test, params)

# run()
# run_keras_optmize()
