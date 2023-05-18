from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)


def eval_metrics(actual, pred):
    f1 = f1_score(actual, pred)
    roc = roc_auc_score(actual, pred)
    rec = recall_score(actual, pred)
    pre = precision_score(actual, pred)
    acc = accuracy_score(actual, pred)
    print("F1-Score:", f1)
    print(confusion_matrix(actual, pred))
    return f1, roc, rec, pre, acc
