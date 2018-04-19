
import os

import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from get_features import get_features


class DefinitionModel:

    scoring_method = {
        "classification": "accuracy",
        "regression": "r2"
    }

    def __init__(self, year, model, model_type, path_data, path_model):
        self.year = year
        self.model = model
        self.model_type = model_type
        self.path_data = path_data
        self.path_model = path_model

        self.data = None
        self.validator = None
        self.prediction = None

    def __str__(self):
        s = "DefinitionModel class:\n"
        s += "Year: {0}\n".format(self.year)
        s += "Model: {0}\n".format(self.get_model())
        s += "Model type: {0}\n".format(self.model_type)
        s += "Path data: {0}\n".format(self.path_data)
        s += "Path model: {0}\n".format(self.path_model)
        return s

    def get_year(self):
        return self.year

    def get_model(self):
        return self.model

    def get_model_type(self):
        return self.model_type

    def get_path_data(self):
        return os.path.join(self.path_data, self.year+".csv")

    def get_path_model(self):
        path = os.path.join(self.path_model, self.model_type)
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path, self.year)
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, self.model+".pkl")

    def get_x(self):
        return self.data.loc[:, get_features("features")]

    def get_y(self):
        if self.model_type == "regression":
            return self.data.diff_points
        elif self.model_type == "classification":
            return self.data.label

    def get_prediction(self):
        return self.prediction

    def set_data(self, data):
        self.data = data

    def set_model_type(self, model_type):
        self.model_type = model_type

    def set_prediction(self, prediction):
        self.prediction = prediction

    def get_validator(self):
        return self.validator

    def load_data(self):
        self.data = pd.read_csv(self.get_path_data(), header=0, sep=",")

    def define_validator(self):
        self.validator = GridSearchCV(self.estimator, self.param_grid, cv=4,
                                      scoring=self.scoring_method[self.model_type])

    def fit_validator(self):
        self.validator.fit(self.get_x(), self.get_y())

    def transform_model(self):
        self.prediction = self.validator.predict(self.get_x())

    def save_model(self):
        if os.path.isfile(self.get_path_model()):
            os.remove(self.get_path_model())
        joblib.dump(self.validator, self.get_path_model())

    def load_model(self):
        self.validator = joblib.load(self.get_path_model())

    def evaluate_classification(self):
        return accuracy_score(self.get_y(), self.transform_model(), normalize=True)
