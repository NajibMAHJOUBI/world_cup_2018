
from itertools import combinations


def get_stacking_approach():
    classification_models = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron",
                             "one_vs_rest"]
    regression_models = ["linear_regression", "decision_tree", "random_forest", "gbt_regressor"]

    available_models = [("classification", model) for model in classification_models]
    available_models += [("regression", model) for model in regression_models]

    approaches = []
    for n_permutation in range(2, len(available_models)+1):
        approaches += list(combinations(available_models, n_permutation))

    n_zfill = len(str(len(approaches)))
    return {"stacking_{0}".format(str(index+1).zfill(n_zfill)): approach for index, approach in enumerate(approaches)}
