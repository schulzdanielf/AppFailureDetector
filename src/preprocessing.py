from cuml.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import cudf as cd


def preprocess(filtering, scaler, smote, x_train, x_test, y_train):
    if filtering == "True":
        print("Filtering")
        with open("data/params/features.pkl", "rb") as inp:
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

        if isinstance(x_train, cd.DataFrame):
            x_train, y_train = x_train.to_pandas(), y_train.to_pandas()

        with open("data/params/smote.pkl", "rb") as inp:
            samp_strat = pickle.load(inp)

        smote = SMOTE(random_state=42, sampling_strategy=samp_strat)
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
