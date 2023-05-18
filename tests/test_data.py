from src.data.data_funcs import get_data, split, train_test

"""
def test_get_data():
    data = get_data()
    assert data.shape == (800000, 301)
    assert data["INDISPONIBILIDADE"].sum() == 3106.0


def test_split():
    mat = get_data()
    train, test = split(0.8, mat)
    assert train.shape == (640000, 301)
    assert test.shape == (160000, 301)


def test_train_test():
    mat = get_data()
    train, test = split(0.8, mat)
    x_train, y_train, x_test, y_test = train_test(train, test)
    assert x_train.shape == (640000, 300)
    assert y_train.shape == (640000, 1)
    assert x_test.shape == (160000, 300)
    assert y_test.shape == (160000, 1)
"""