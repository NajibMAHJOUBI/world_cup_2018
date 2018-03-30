
import os

from pyspark.sql.types import StructType, StructField, StringType, FloatType
from sklearn.metrics import accuracy_score


class AveragingPredictions:

    stages = ["1st_stage", "2nd_stage", "3rd_stage", "4th_stage", "5th_stage", "6th_stage"]

    schema = StructType([
        StructField("matches", StringType(), True),
        StructField("label", FloatType(), True),
        StructField("prediction", FloatType(), True)])

    def __init__(self, spark, year, path_prediction, classification_methods, regression_models,
                 stacking_models, stacking_names):
        self.spark = spark
        self.year = str(year)
        self.path_prediction = path_prediction
        self.classification_models = classification_methods
        self.regression_models = regression_models
        self.stacking_models = stacking_models
        self.stacking_names = stacking_names
        self.accuracy = None

    def __str__(self):
        s = "Year = {0}\n".format(self.year)
        s += "Classification models: {0}".format(self.classification_models)
        return s

    def run(self):
        dic_data = {}
        if self.classification_models is not None:
            dic_data["classification"] = self.merge_unitary_model("classification")
            data = self.average_prediction(dic_data["classification"])
        if self.regression_models is not None:
            dic_data["regression"] = self.merge_unitary_model("regression")
            data = self.average_prediction(dic_data["regression"])
        if self.stacking_models is not None:
            dic_data["stacking"] = self.merge_stacking()
            data = self.average_prediction(dic_data["stacking"])
        self.compute_accuracy(data)

    def get_path_classification(self, stage, model):
        return os.path.join(self.path_prediction, "classification", self.year, stage, model)

    def get_path_regression(self, stage, model):
        return os.path.join(self.path_prediction, "regression", self.year, stage, model)

    def get_path_stacking(self, stage, model, model_name):
        return os.path.join(self.path_prediction, "stacking", self.year, stage, model_name, model)

    def get_path(self, type_, stage, model, model_name=None):
        if type_ == "classification":
            return self.get_path_classification(stage, model)
        elif type_ == "regression":
            return self.get_path_regression(stage, model)
        elif type_ == "stacking":
            return self.get_path_stacking(stage, model, model_name)

    def get_models_names(self, type_):
        if type_ == "classification":
            return self.classification_models
        elif type_ == "regression":
            return self.regression_models
        elif type_ == "stacking":
            return self.stacking_models

    def get_accuracy(self):
        return self.accuracy

    def union_stages(self, type_, model, model_name=None):
        data = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD(), self.schema)
        for stage in self.stages[:]:
            temp = self.spark.read.csv(self.get_path(type_, stage, model, model_name), sep=",", header=True,
                                       schema=self.schema)
            data = data.union(temp)
        return data

    def merge_unitary_model(self, type_):
        models = self.get_models_names(type_)
        data = (self.union_stages(type_, models[0])
                .withColumnRenamed("prediction", "{0}_{1}".format(type_, models[0])))
        for model in models[1:]:
            temp = (self.union_stages(type_, model)
                    .withColumnRenamed("prediction", "{0}_{1}".format(type_, model))
                    .drop("label"))
            data = data.join(temp, on="matches")
        return data.toPandas()

    def merge_stacking(self):
        data = None
        for index_model, model in enumerate(self.get_models_names("stacking")):
            # print("Model: {0}".format(model))
            for index_name, model_name in enumerate(self.stacking_names[:]):
                # print("  Model name: {0}".format(model_name)),
                temp = (self.union_stages("stacking", model, model_name)
                        .withColumnRenamed("prediction", "{0}_{1}".format(model_name, model))).toPandas()
                if index_model == 0 and index_name == 0:
                    data = temp
                else:
                    temp = temp.drop(["label"], axis=1)
                    data = data.merge(temp, on="matches")
        return data

    def merge_methods(self):
        pass

    def average_prediction(self, data):
        features = data.columns.values.tolist()
        features.remove("matches")
        features.remove("label")

        data["combined"] = data.apply(lambda x: [x[column] for column in features], axis=1)
        data["win"] = data.apply(lambda x: x["combined"].count(2.0) / float(len(x["combined"])), axis=1)
        data["losse"] = data.apply(lambda x: x["combined"].count(1.0) / float(len(x["combined"])), axis=1)
        data["draw"] = data.apply(lambda x: x["combined"].count(0.0) / float(len(x["combined"])), axis=1)

        def prediction_probability(p_win, p_losse, p_draw):
            proba = [p_win, p_losse, p_draw]
            ind = proba.index(max(proba))
            if ind == 0:
                return 2.0
            elif ind == 1:
                return 1.0
            elif ind == 2:
                return 0.0

        data["prediction"] = data.apply(lambda x: prediction_probability(x["win"], x["losse"], x["draw"]), axis=1)

        return data[["matches", "label", "prediction"]]

    def compute_accuracy(self, data):
        data_list = data.to_dict("list")
        label = data_list["label"]
        prediction = data_list["prediction"]
        self.accuracy = accuracy_score(label, prediction, normalize=True)



if __name__ == "__main__":
    from get_spark_session import get_spark_session
    from get_classification_approach import get_classification_approach
    from get_regression_approach import get_regression_approach
    from get_stacking_approach import get_stacking_approach
    spark = get_spark_session("Classification Model")

    # average_predictions = AveragingPredictions(spark, 2014, "./test/prediction",
    #                                            None, None, ["decision_tree"], sorted(get_stacking_approach().keys()))
    # get_classification_approach(), get_regression_approach(), ["logistic_regression"], get_stacking_approach().keys())

    for model_name in sorted(get_stacking_approach().keys()):
        average_predictions = AveragingPredictions(spark, 2014, "./test/prediction",
                                                   None, None, ["logistic_regression"], [model_name])
        average_predictions.run()
        print("Stacking - LR {0}: {1}".format(model_name, average_predictions.get_accuracy()))
