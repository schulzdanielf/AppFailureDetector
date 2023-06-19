from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import pickle
from src.data.data_funcs import split, train_test
import pandas as pd
import numpy as np


def get_data():
    mat = pd.read_csv("data/raw/matomo.csv", dtype=np.float32)

    return mat


mat = get_data()

train, test = split(0.75, mat)

x_train, y_train, x_test, y_test = train_test(train, test)


def filter(x_train, x_test):
    print("Filtering")
    with open("data/params/features.pkl", "rb") as inp:
        features = pickle.load(inp)
    x_train = x_train[features]
    x_test = x_test[features]

    return x_train, x_test


x_train, x_test = filter(x_train, x_test)

model = RandomForestClassifier(n_estimators=100)

model.fit(x_train, y_train)

# Salvar o modelo em disco
filename = 'models/rf_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
