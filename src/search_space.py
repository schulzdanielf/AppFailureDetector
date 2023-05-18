from hyperopt import hp


def get_search_space():
    search_space = hp.choice(
        "classifier_type",
        [
            {
                "type": "rf",
                "n_estimators": hp.choice(
                    "n_estimators", [25, 50, 100, 200, 500, 1000]
                ),
                "preprocessing": hp.choice(
                    "p_rf",
                    [
                        "scaler",
                        "filter",
                        "all",
                        "none",
                        "fi_ss",
                        "fi_sm",
                        "ss_sm",
                        "smote",
                    ],
                ),
            },
            {
                "type": "knn",
                "n_neighbors": hp.choice("n_neighbors", [1, 3, 5, 7, 9]),
                "metric": hp.choice("metric", ["euclidean", "manhattan", "minkowski"]),
                "preprocessing": hp.choice(
                    "p_knn",
                    [
                        "scaler",
                        "filter",
                        "all",
                        "none",
                        "fi_ss",
                        "fi_sm",
                        "ss_sm",
                        "smote",
                    ],
                ),
            },
            {
                "type": "svm",
                "C": hp.choice("C", [0.01, 0.1, 0.5, 1, 10, 100]),
                "kernel": hp.choice("kernel", ["rbf", "sigmoid"]),
                "preprocessing": hp.choice(
                    "p_svm",
                    [
                        "scaler",
                        "filter",
                        "all",
                        "none",
                        "fi_ss",
                        "fi_sm",
                        "ss_sm",
                        "smote",
                    ],
                ),
            },
        ],
    )

    search_space_lstm = hp.choice(
        "classifier_type",
        [
            {
                "type": "lstm",
                "activation": hp.choice("activation", ["relu", "tanh"]),
                "units": hp.choice("units", [32, 64, 128, 256, 512, 1024, 2048]),
                "batch": hp.choice("batch", [256, 512, 1024, 2048]),
                "dropout": hp.choice("dropout", [True, False]),
                "learning_rate": hp.loguniform("learning_rate", 1e-4, 1e-2),
                "preprocessing": hp.choice(
                    "p_lstm",
                    [
                        "scaler",
                        "filter",
                        "all",
                        "none",
                        "fi_ss",
                        "fi_sm",
                        "ss_sm",
                        "smote",
                    ],
                ),
            }
        ],
    )

    search_space_cpu = hp.choice(
        "classifier_type",
        [
            {
                "type": "nb",
                "var_smoothing": hp.choice("var_smoothing", [1e-9, 1e-5, 1e-20]),
                "preprocessing": hp.choice(
                    "p_nb",
                    [
                        "scaler",
                        "filter",
                        "all",
                        "none",
                        "fi_ss",
                        "fi_sm",
                        "ss_sm",
                        "smote",
                    ],
                ),
            },
            {
                "type": "dt",
                "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"]),
                "preprocessing": hp.choice(
                    "p_dt",
                    [
                        "scaler",
                        "filter",
                        "all",
                        "none",
                        "fi_ss",
                        "fi_sm",
                        "ss_sm",
                        "smote",
                    ],
                ),
            },
            {
                "type": "ada",
                "n_estimators": hp.choice("n_estimators_ada", [50, 25, 100]),
                "learning_rate": hp.choice("learning_rate", [1, 0.5, 2]),
                "preprocessing": hp.choice(
                    "p_ada",
                    [
                        "scaler",
                        "filter",
                        "all",
                        "none",
                        "fi_ss",
                        "fi_sm",
                        "ss_sm",
                        "smote",
                    ],
                ),
            },
        ],
    )

    return search_space, search_space_cpu, search_space_lstm
