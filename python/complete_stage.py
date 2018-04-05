
import os
import pandas as pd

from featurization_data import FeaturizationData
from model_classification import ClassificationModel
from model_regression import RegressionModel


class CompleteStage:

    def __init__(self, year, model, model_method, stage, path_model, path_prediction):
        self.year = year
        self.model_name = model
        self.model_method = model_method
        self.stage = stage
        self.path_model = path_model
        self.path_prediction = path_prediction

        self.prediction = None
        self.model_class = None

    def __str__(self):
        s = "Year: {0}\n".format(self.year)
        s += "Model: {0}\n".format(self.model_name)
        s += "Model method: {0}\n".format(self.model_method)
        s += "Stage: {0}\n".format(self.stage)
        s += "Path model: {0}\n".format(self.path_model)
        s += "Path prediction: {0}\n".format(self.path_prediction)
        return s

    def run(self):
        self.load_model_class()
        self.load_data()
        self.load_model()
        self.compute_prediction()
        self.save_prediction()

    def get_prediction(self):
        return self.prediction

    def get_path_prediction(self):
        path = os.path.join(self.path_prediction, self.model_method)
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path, self.year)
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path, self.stage)
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, self.model_name+".csv")

    def load_data(self):
        featurization_data = FeaturizationData(self.year, ["WCF"], None, stage=self.stage)
        featurization_data.set_dates()
        self.model_class.set_data(featurization_data.compute_data_confederation("WCF"))

    def load_model_class(self):
        if self.model_method == "classification":
            self.model_class = ClassificationModel(self.year, self.model_name, None, self.path_model)
        elif self.model_method == "regression":
            self.model_class = RegressionModel(self.year, self.model_name, None, self.path_model)

    def load_model(self):
        self.model_class.load_model()

    def compute_prediction(self):
        self.prediction = self.model_class.transform_model()

    def define_label_prediction(self):
        return pd.DataFrame({"label": self.model_class.get_y(),
                             "prediction": self.prediction})

    def save_prediction(self):
        self.define_label_prediction().to_csv(self.get_path_prediction(), header=True, index=False)


if __name__ == "__main__":
    import sys
    sys.path.append("./pyspark")
    from get_classification_models import get_classification_models
    from get_regression_models import get_regression_models
    from get_competition_dates import get_competition_dates

    dic_models = {
        "classification": get_classification_models(),
        "regression": get_regression_models()
    }
    year = "2014"
    for stage in get_competition_dates(year).keys():
        for model_method in dic_models.keys():
            for model in dic_models[model_method]:
                complete_stage = CompleteStage(year, model, model_method, stage,
                                               "./test/sklearn/model",
                                               "./test/sklearn/prediction")
                complete_stage.run()
