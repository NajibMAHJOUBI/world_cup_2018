import numpy as np
import os
import pandas as pd

from featurization_data import FeaturizationData
from model_definition import DefinitionModel


def get_prediction(x):
    threshold = 0.25
    if float(np.abs(x)) <= threshold:
        return 0.0
    elif x > threshold:
        return 2.0
    elif x < -1.0 * threshold:
        return 1.0


class CompleteStage(DefinitionModel):

    def __init__(self, year, model, model_method, stage, path_model, path_prediction):
        DefinitionModel.__init__(self, year, model, model_method, None, path_model)
        self.stage = stage
        # self.path_model = path_model
        self.path_prediction = path_prediction

        self.data = None
        self.prediction = None

    def __str__(self):
        s = "Year: {0}\n".format(self.get_year())
        s += "Model: {0}\n".format(self.get_model())
        s += "Model method: {0}\n".format(self.get_model_type())
        s += "Stage: {0}\n".format(self.stage)
        s += "Path model: {0}\n".format(self.path_model)
        s += "Path prediction: {0}\n".format(self.path_prediction)
        s += "ModelDefinition: {0}\n".format(DefinitionModel.__str__(self))
        return s

    def run(self):
        self.load_data()
        self.load_model()
        self.compute_prediction()
        self.save_prediction()

    def get_prediction(self):
        return self.prediction

    def get_path_prediction(self):
        path = os.path.join(self.path_prediction, self.get_model_type())
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path, self.get_year())
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(path, self.stage)
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, self.get_model()+".csv")

    def load_data(self):
        features_data = FeaturizationData(self.year, ["WCF"], None, stage=self.stage)
        features_data.set_dates()
        self.data = features_data.compute_data_confederation("WCF")

    def compute_prediction(self):
        self.set_data(self.data)
        self.prediction = self.transform_model()

    def define_label_prediction(self):
        if self.model_type == "classification":
            label_prediction = {"label": self.get_y(),
                                "prediction": self.get_prediction()}
        elif self.model_type == "regression":
            label_prediction = {}
            self.set_model_type("classification")
            label_prediction.update({"label": self.get_y()})
            self.set_model_type("regression")
            label_prediction.update({"diff_points_label": self.get_y(),
                                     "diff_points_prediction": self.get_prediction(),
                                     "prediction":
                                         np.array([get_prediction(diff_points)
                                                   for diff_points in self.get_prediction().tolist()])})

        return pd.DataFrame(label_prediction)

    def save_prediction(self):
        self.define_label_prediction().to_csv(self.get_path_prediction(), header=True, index=False)


if __name__ == "__main__":
    import sys
    from get_classification_models import get_classification_models
    from get_regression_models import get_regression_models
    sys.path.append("./pyspark")
    from get_competition_dates import get_competition_dates

    dic_models = {
        "classification": get_classification_models(["gaussian_classifier"]),
        "regression": get_regression_models()
    }
    for year in ["2006", "2010", "2014"]:
        for stage in get_competition_dates(year).keys():
            for model_type in dic_models.keys():
                for model in dic_models[model_type]:
                    complete_stage = CompleteStage(year, model, model_type, stage,
                                                   "./test/sklearn/model",
                                                   "./test/sklearn/prediction")
                    # print(complete_stage)
                    complete_stage.run()

