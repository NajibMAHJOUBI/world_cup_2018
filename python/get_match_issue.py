import numpy as np


def get_match_issue(x):
    threshold = 0.25
    if float(np.abs(x)) <= threshold:
        return 0.0
    elif x > threshold:
        return 2.0
    elif x < -1.0 * threshold:
        return 1.0
