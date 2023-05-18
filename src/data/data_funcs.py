import cupy as cp
import cudf as cd


def split(r, mat):
    i = int(len(mat) * r)

    return mat[:i], mat[i:]


def get_data():
    mat = cd.read_csv("data/raw/matomo.csv", dtype=cp.float32)

    return mat


def train_test(train, test):
    x_train = train.drop(["INDISPONIBILIDADE"], axis=1)
    y_train = train[["INDISPONIBILIDADE"]]

    x_test = test.drop(["INDISPONIBILIDADE"], axis=1)
    y_test = test[["INDISPONIBILIDADE"]]

    return x_train, y_train, x_test, y_test
