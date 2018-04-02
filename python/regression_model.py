import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import Lars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from get_features import get_features
from get_match_issue import get_match_issue


class RegressionModel:

    def __init__(self, year, model_regression, path_data, path_model):
        self.year = year
        self.model_regression = model_regression
        self.path_data = path_data
        self.path_model = path_model

        self.estimator = None
        self.param_grid = None
        self.validator = None

    def __str__(self):
        pass

    def run(self):
        self.load_data()
        self.define_estimator()
        self.define_grid_parameters()
        self.define_validator()
        self.fit_validator()
        self.save_model()

    def get_path_data(self):
        return os.path.join(self.path_data, self.year+".csv")

    def get_path_model(self):
        return os.path.join(self.path_model, "regression", self.year, self.model_regression+".pkl")

    def get_x(self):
        return self.data.loc[:, get_features("features")]

    def get_y(self):
        return self.data.diff_points

    def load_data(self):
        self.data = pd.read_csv(self.get_path_data(), header=0, sep=",")

    def define_estimator(self):
        if self.model_regression == "linear_regression":
            self.estimator = LinearRegression()
        elif self.model_regression == "ridge":
            self.estimator = Ridge()
        elif self.model_regression == "lars":
            self.estimator = Lars()

    def define_grid_parameters(self):
        if self.model_regression == "linear_regression":
            self.param_grid = [{'fit_intercept': [True, False],
                                'normalize': [True, False]}]
        elif self.model_regression == "ridge":
            self.param_grid = [{'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                                'fit_intercept': [True, False],
                                'normalize': [True, False],
                                'solver': ['svd', 'cholesky', 'lsqr',
                                           'sparse_cg', 'sag', 'saga']}]
        elif self.model_regression == "lars":
            self.param_grid = [{'fit_intercept': [True, False],
                                'normalize': [True, False],
                                'precompute': [True, False],
                                'positive': [True, False]}]

    def define_validator(self):
        self.validator = GridSearchCV(self.estimator, self.param_grid, cv=4, scoring='r2')

    def fit_validator(self):
        self.validator.fit(self.get_x(), self.get_y())

    def transform_model(self):
        return self.validator.predict(self.get_x())

    def load_model(self):
        self.validator = joblib.load(self.get_path_model())

    def save_model(self):
        if not os.path.join(self.path_model, "regression"):
            os.makedirs(os.path.join(self.path_model, "regression"))
        if not os.path.isdir(os.path.join(self.path_model, "regression", self.year)):
            os.makedirs(os.path.join(self.path_model, "regression", self.year))
        if os.path.isfile(self.get_path_model()):
            os.remove(self.get_path_model())
        joblib.dump(self.validator, self.get_path_model())

    def evaluate(self, model_):
        if model_ == "classification":
            return accuracy_score(self.get_y(), np.array(map(lambda x: get_match_issue(x), self.transform_model())),
                                  normalize=True)
        elif model_ == "regression":
            return r2_score(self.get_y(), self.transform_model())


if __name__ == "__main__":
    models = ["linear_regression", "ridge", "lars"]
    dic_regression, dic_classification = {}, {}
    for model in models:
        print("Model: {0}".format(model))
        regression_model = RegressionModel("2014", model,
                                           "./test/sklearn/training",
                                           "./test/sklearn/model")
        regression_model.run()
        # print(type(regression_model.transform_model()))
        # print(regression_model.transform_model().shape)
        dic_regression[model] = regression_model.evaluate("regression")
        dic_classification[model] = regression_model.evaluate("classification")

    print("Regression:")
    for item in dic_regression.iteritems():
        print(item)
    print("")
    print("Classification:")
    for item in dic_classification.iteritems():
        print(item)
