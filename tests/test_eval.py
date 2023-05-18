from src.eval import eval_metrics


def test_eval_metrics():

    assert eval_metrics([1, 0], [1, 0])[0] == 1
    assert len(eval_metrics([1, 0], [1, 0])) == 5