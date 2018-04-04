

from featurization_data import FeaturizationData
from classification_model import ClassificationModel
from regression_model import RegressionModel


class CompleteStage:

    def __init__(self, year, model, model_method, stage, path_model, path_prediction):
        self.year = year
        self.model = model
        self.model_method = model_method
        self.stage = stage
        self.path_model = path_model
        self.path_prediction = path_prediction

        self.data = None

    def __str__(self):
        s = "Year: {0}\n".format(self.year)
        s += "Model: {0}\n".format(self.model)
        s += "Model method: {0}\n".format(self.model_method)
        s += "Stage: {0}\n".format(self.stage)
        s += "Path model: {0}\n".format(self.path_model)
        s += "Path prediction: {0}\n".format(self.path_prediction)
        return s

    def run(self):
        self.load_data()
        self.load_model()

    def load_data(self):
        featurization_data = FeaturizationData(self.year, ["WCF"], None, stage=self.stage)
        featurization_data.set_dates()
        self.data = featurization_data.compute_data_confederation("WCF")

    def load_model(self):
        if self.model_method == "classification":
            classification_model = ClassificationModel(self.year, self.model, None, self.path_model)
            classification_model.load_model()
            self.model = classification_model.get_validator()
        elif self.model_method == "regression":
            regression_model = RegressionModel(self.year, self.model, None, self.path_model)
            regression_model.load_model()
            self.model = regression_model.get_validator()


if __name__ == "__main__":
    complete_stage = CompleteStage("2014", "logistic_regression", "classification", "1st_stage", "./test/sklearn/model", None)
    complete_stage.run()
