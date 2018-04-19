import os
from model_definition import DefinitionModel


class AccuracyModel(DefinitionModel):

    def __init__(self, year, model, model_type_, path_prediction):
        DefinitionModel.__init__(self, year, None, "classification", None, None)

        self.model_type_ = model_type_
        self.path_prediction = path_prediction

    def __str__(self):
        pass

    def get_path_prediction(self, stage):
        return os.path.join(self.path_prediction, self.model_type_, self.get_year(), stage, model+'.csv')

    def append_stages(self):
        pass

    def save_accuracy(self):
        pass
