import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from keras.layers import LSTM, Dense, Dropout
import keras.backend as K
import warnings
import pandas as pd
import mlflow
from src.preprocessing import preprocessing
from src.data.data_funcs import get_data
from sklearn.metrics import f1_score, confusion_matrix


warnings.filterwarnings("ignore")


search_space_lstm = hp.choice(
    "classifier_type",
    [
        {
            "type": "lstm",
            "activation": hp.choice("activation", ["relu", "tanh"]),
            "units": hp.quniform("units", 32, 2048, 32),
            "batch": hp.choice("batch", [2048]),
            "dropout": hp.choice("dropout", [True, False]),
            "learning_rate": hp.loguniform(
                "learning_rate", np.log(0.000001), np.log(0.001)
            ),
            "preprocessing": hp.choice(
                "p_lstm",
                ["scaler", "filter", "all", "none", "fi_ss", "fi_sm", "ss_sm", "smote"],
            ),
        }
    ],
)


def f_0(y):
    if y["INDISPONIBILIDADE"] == 0:
        val = 1
    else:
        val = 0
    return val


def f_1(y):
    if y["INDISPONIBILIDADE"] == 1:
        val = 1
    else:
        val = 0
    return val


def ajusta_y(y):
    y["0"] = y.apply(f_0, axis=1)
    y["1"] = y.apply(f_1, axis=1)
    y = y[["0", "1"]]
    return y


# def create_sequences(values, time_steps=1):
#    output = []
#    for i in range(len(values) - time_steps + 1):
#        output.append(values[i : (i + time_steps)])
#    return np.stack(output)


def create_sequences(values, time_steps=1):
    return np.asarray(
        [values[i : (i + time_steps)] for i in range(len(values) - time_steps + 1)]
    )


def ajusta_y_timestep(y, time_steps=1):
    new_y = y[time_steps - 1 :]
    return new_y


def transform_dimension_timesteps(train_x, train_y, time_steps=1):
    train_x = create_sequences(train_x, time_steps)
    train_y = ajusta_y_timestep(train_y, time_steps)
    train_y = train_y.values.reshape(-1, 2)

    print(train_y.shape)
    return train_x, train_y


def ajusta_saida(y_pred):
    y_pred_c = []
    for x in y_pred:
        y_pred_c.append(np.argmax(x))
    return y_pred_c


"""def objective_lstm(params):
    units = params['units']
    model = keras.Sequential()
    model.add(LSTM(params['units'], activation=params['activation'],
                   return_sequences=False, input_shape=(1, shape)))

    model.add(Dense(params['units'], activation=params['activation'],
                    input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=params['epochs'],
              batch_size=params['batch_size'], verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    return -score[1]
"""


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


class MyModel:
    def build(self, **kwargs):
        activation = kwargs.get("activation")
        shape = kwargs.get("shape")
        # batch = int(kwargs.get("batch"))
        dropout = kwargs.get("dropout")
        lr = kwargs.get("learning_rate")
        units = float(kwargs.get("units"))
        units = int(units)

        model = keras.Sequential()
        model.add(
            LSTM(
                units,
                activation=activation,
                return_sequences=False,
                input_shape=(1, shape),
            )
        )
        model.add(Dense(units, activation=activation))
        if dropout:
            model.add(Dropout(rate=0.2))
        model.add(Dense(units, activation=activation))
        model.add(Dense(units, activation=activation))
        model.add(Dense(2, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer=RMSprop(learning_rate=lr),
            metrics=[get_f1],
        )
        return model

    def fit(self, model, *args, **kwargs):
        return model.fit(*args, batch_size=batch, **kwargs, verbose=0)


def predict_keras(model, test_x):
    x = create_sequences(test_x.copy(), 1)
    predicted = model.predict(x)
    predicted = ajusta_saida(predicted)
    return predicted


def split(r, mat):
    i = int(len(mat) * r)

    return mat[:i], mat[i:]


def train_test(train, test):
    x_train = train.drop(["INDISPONIBILIDADE"], axis=1)
    y_train = train[["INDISPONIBILIDADE"]]

    x_test = test.drop(["INDISPONIBILIDADE"], axis=1)
    y_test = test[["INDISPONIBILIDADE"]]

    return x_train, y_train, x_test, y_test


def test_model(model, batch, epochs, x_train, y_train, x_val, y_val):
    print("Training the model")
    model.fit(x_train, y_train, batch_size=batch, epochs=epochs)

    pred = predict_keras(model, x_val)
    f1 = f1_score(y_val, pred)
    print(confusion_matrix(y_val, pred))

    return f1


def objective_keras(params):
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        x, y = data
        p = params["preprocessing"]
        print("Preprocess:", p)
        df = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        train, test = split(0.75, df)
        x_train, y_train, x_val, y_val = train_test(train, test)

        x_train, y_train, x_val = preprocessing(p, x_train, y_train, x_val)
        print("Converting training data")
        y_train = ajusta_y(y_train)
        x_train, y_train = transform_dimension_timesteps(x_train, y_train, time_steps=1)

        del params["preprocessing"]
        del params["type"]
        mlflow.log_param("model", "lstm")
        mlflow.log_param("model_selection", "train_test")
        mlflow.log_param("stage", "tuning")

        params["shape"] = x_train.shape[2]
        clf = MyModel().build(**params)
        batch, epochs = int(params["batch"]), 40

        print(params)

        f1 = test_model(clf, batch, epochs, x_train, y_train, x_val, y_val)

        f1 = f1 + test_model(clf, batch, epochs, x_train, y_train, x_val, y_val)

        f1 = f1 + test_model(clf, batch, epochs, x_train, y_train, x_val, y_val)

        f1 = f1 / 3

        print("Média F1-SCORE", f1)
        mlflow.log_metric("f1_val", f1)

        # Because fmin() tries to minimize the objective,
        # this function must return the negative accuracy.
        return {"loss": -f1, "status": STATUS_OK}


def find_best(x, y, evals, space):
    global data
    data = [x, y]
    rstate = np.random.default_rng(42)
    trials = Trials()
    best_result = fmin(
        fn=objective_keras,
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

    print(result)

    return result, trials, key


def run():
    print("Reading data")
    mat = get_data()
    print("Spliting the data into train/test with 75/25 proportion")
    train, test = split(0.75, mat)
    print("Spliting the data into x and y features")
    x_train, y_train, x_test, y_test = train_test(train, test)

    print("Find best parameters for LSTM model")
    find_best(x_train, y_train, 100, search_space_lstm)
