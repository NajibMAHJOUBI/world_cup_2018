
from sklearn.metrics import accuracy_score


def get_accuracy(label, prediction):
    return accuracy_score(label, prediction, normalize=True)
