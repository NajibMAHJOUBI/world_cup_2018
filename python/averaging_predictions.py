
import os

from pyspark.sql.types import StructType, StructField, StringType, FloatType

from get_spark_session import get_spark_session
from get_classification_approach import get_classification_approach
from get_regression_approach import get_regression_approach


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

    def __str__(self):
        s = "Year = {0}\n".format(self.year)
        s += "Classification models: {0}".format(self.classification_models)
        return s

    def run(self):
        dic_data = {}
        if self.classification_models is not None:
            dic_data["classification"] = self.merge_unitary_model("classification")
        if self.regression_models is not None:
            dic_data["regression"] = self.merge_unitary_model("regression")

    def get_path_classification(self, stage, model):
        return os.path.join(self.path_prediction, "classification", self.year, stage, model)

    def get_path_regression(self, stage, model):
        return os.path.join(self.path_prediction, "regression", self.year, stage, model)

    def get_path(self, type_, stage, model):
        if type_ == "classification":
            return self.get_path_classification(stage, model)
        elif type_ == "regression":
            return self.get_path_regression(stage, model)

    def get_path_stacking(self):
        pass

    def get_models_names(self, type_):
        if type_ == "classification":
            return self.classification_models
        elif type_ == "regression":
            return self.regression_models

    def union_stages(self, type_, model):
        data = self.spark.read.csv(self.get_path(type_, self.stages[0], model), sep=",", header=True,
                                   schema=self.schema)
        for stage in self.stages[1:]:
            temp = self.spark.read.csv(self.get_path(type_, stage, model), sep=",", header=True,
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
        return data

    def average_prediction(self, data):
        features = data.columns
        features.remove("matches")
        features.remove("label")
        features_broadcast = self.spark.sparkContext.broadcast(features)
        features_number = self.spark.sparkContext.broadcast(float(len(features)))

        data = (data.rdd
                .map(lambda x: (x["matches"], x["label"], [x[feature] for feature in features_broadcast.value]))
                .map(lambda x: (x[0], x[1], x[2].count(2.0)/features_number.value,
                                            x[2].count(1.0)/features_number.value,
                                            x[2].count(0.0)/features_number.value)))

        return self.spark.createDataFrame(data, ["matches", "label", "win", "losse", "drawn"])


if __name__ == "__main__":
    spark = get_spark_session("Classification Model")

    average_predictions = AveragingPredictions(spark, 2014, "./test/prediction",
                                               get_classification_approach(), get_regression_approach())
    # print(average_predictions)
    average_predictions.run()
