
import os
import pandas as pd
from get_features import get_features
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score


class DefinitionModel:

    def __init__(self, year, model, model_type, path_data, path_model):
        self.year = year
        self.model = model
        self.model_type = model_type
        self.path_data = path_data
        self.path_model = path_model

    def __str__(self):
        pass

    def get_path_data(self):
        return os.path.join(self.path_data, self.year+".csv")

    def get_path_model(self):
        path = os.path.join(self.path_model, self.model_type)
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path, self.year)
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, self.model_classifier+".pkl")

    def get_x(self):
        return self.data.loc[:, get_features("features")]

    def get_y(self):
        return self.data.diff_points

    def set_data(self, data):
        self.data = data

    def get_validator(self):
        return self.validator

    def load_data(self):
        self.data = pd.read_csv(self.get_path_data(), header=0, sep=",")

    def define_validator(self):
        self.validator = GridSearchCV(self.estimator, self.param_grid, cv=4, scoring='accuracy')

    def fit_validator(self):
        self.validator.fit(self.get_x(), self.get_y())

    def transform_model(self):
        return self.validator.predict(self.get_x())

    def save_model(self):
        if os.path.isfile(self.get_path_model()):
            os.remove(self.get_path_model())
        joblib.dump(self.validator, self.get_path_model())

    def load_model(self):
        self.validator = joblib.load(self.get_path_model())

    def evaluate_classification(self):
        return accuracy_score(self.get_y(), self.transform_model(), normalize=True)
