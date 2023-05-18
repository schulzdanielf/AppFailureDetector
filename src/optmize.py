from sklearn.model_selection import RandomizedSearchCV
import json


def best_knn(x, y):
    from sklearn.neighbors import KNeighborsClassifier

    distributions = {"n_neighbors": [3, 5, 7], "p": [1, 2]}
    model = RandomizedSearchCV(
        KNeighborsClassifier(n_jobs=6),
        distributions,
        random_state=0,
        scoring="f1",
        cv=4,
        n_jobs=2,
    )
    search = model.fit(x, y)
    search.best_params_
    print(search.best_params_)
    update_hyper(search.best_params_, "knn")


def best_rf(x, y):
    from sklearn.ensemble import RandomForestClassifier

    distributions = {"n_estimators": [50, 200], "min_samples_split": [2]}
    model = RandomizedSearchCV(
        RandomForestClassifier(), distributions, random_state=0, scoring="f1", cv=4
    )
    search = model.fit(x, y)
    search.best_params_
    print(search.best_params_)
    update_hyper(search.best_params_, "rf")


def best_nb(x, y):
    print("Initiating Naive Bayes tunning")
    from sklearn.naive_bayes import GaussianNB

    distributions = {"var_smoothing": [1e-09]}
    model = RandomizedSearchCV(
        GaussianNB(), distributions, random_state=0, scoring="f1", cv=4
    )
    search = model.fit(x, y)
    print("Best score", search.best_score_)
    print("Best params", search.best_params_)
    update_hyper(search.best_params_, "nb")


def best_svm(x, y):
    print("Initiating Support Vector Machine tunning")
    from sklearn.svm import SVC

    distributions = {
        "kernel": ["linear", "rbf", "poly"],
        "gama": ["scale"],
        "C": [1.0],
        "degree": [3],
    }
    model = RandomizedSearchCV(SVC(), distributions, random_state=0, scoring="f1", cv=4)
    search = model.fit(x, y)
    print("Best score", search.best_score_)
    print("Best params", search.best_params_)
    update_hyper(search.best_params_, "svm")


def best_dt(x, y):
    print("Initiating Decision Tree tunning")
    from sklearn.tree import DecisionTreeClassifier

    distributions = {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    }
    model = RandomizedSearchCV(
        DecisionTreeClassifier(), distributions, random_state=0, scoring="f1", cv=4
    )
    search = model.fit(x, y)
    print("Best score", search.best_score_)
    print("Best params", search.best_params_)
    update_hyper(search.best_params_, "dt")


def best_ada(x, y):
    print("Initiating Adaboost tunning")
    from sklearn.ensemble import AdaBoostClassifier

    distributions = {"n_estimators": [50, 25, 100], "learning_rate": [1, 0.5]}
    model = RandomizedSearchCV(
        AdaBoostClassifier(),
        distributions,
        random_state=0,
        scoring="f1",
        cv=4,
        n_jobs=2,
    )
    search = model.fit(x, y)
    print("Best score", search.best_score_)
    print("Best params", search.best_params_)
    update_hyper(search.best_params_, "ada")


def update_hyper(dict, key):
    f = open("/app/params/best_hyper.json")
    data = json.load(f)
    f.close()
    data[key] = dict
    print(data)
    with open("/app/params/best_hyper.json", "w") as fp:
        json.dump(data, fp)
