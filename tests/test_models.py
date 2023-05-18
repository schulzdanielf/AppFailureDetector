from src.models.classifiers import get_model


def test_get_model():
    clf = get_model("rf", {"n_estimators": 100})
    assert clf.n_estimators == 100

    clf = get_model("svm", {"C": 1.0})
    assert clf.C == 1.0

    clf = get_model("knn", {"n_neighbors": 5})
    assert clf.n_neighbors == 5

    clf = get_model("dt", {"max_depth": 5})
    assert clf.max_depth == 5

    clf = get_model("lstm", {"epochs": 5, "batch_size": 32,
                             "shape": 300, "units": 32,
                             "dropout": True, "learning_rate": 0.001,
                             "activation": "relu", "preprocessing": "scaler",
                             "batch": 2048})
    assert len(clf.layers) == 6