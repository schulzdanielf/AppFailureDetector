from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from cuml.svm import SVC
from cuml.neighbors import KNeighborsClassifier
from cuml.ensemble import RandomForestClassifier
from src.keras_optmize import MyModel


def get_model(classifier_type, params, shape=0):
    if classifier_type == "knn":
        clf = KNeighborsClassifier(**params)
    elif classifier_type == "svm":
        clf = SVC(**params)
    elif classifier_type == "nb":
        clf = GaussianNB(**params)
    elif classifier_type == "rf":
        clf = RandomForestClassifier(**params)
    elif classifier_type == "ada":
        clf = AdaBoostClassifier(**params)
    elif classifier_type == "nb":
        clf = GaussianNB(**params)
    elif classifier_type == "dt":
        clf = DecisionTreeClassifier(**params)
    elif classifier_type == "lstm":
        params["shape"] = shape
        clf = MyModel().build(**params)
    else:
        return 0

    return clf
