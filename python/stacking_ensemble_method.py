import os
from classification_model import ClassificationModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf
from get_data_schema import get_data_schema
from get_spark_session import get_spark_session


class Stacking:
    def __init__(self, spark_session, year, stage, available_classifier, stacking_classifier, validator, path_transform,
                 path_model):
        self.spark = spark_session
        self.year = year
        self.stage = stage
        self.classifier_available = available_classifier
        self.classifier_stacking = stacking_classifier
        self.validator = validator
        self.path_transform = path_transform  # "./test/transform"
        self.path_model = path_model

        self.data = None
        self.new_data = None
        self.evaluate = None
        self.stacking_model = None

    def __str__(self):
        s = "Stacking model\n"
        s += "Available classifier: {0}".format(self.classifier_available)
        s += "Stacking classifier: {0}".format(self.stacking_classifier)
        return s

    def run(self):
        self.merge_all_data()
        self.create_features()
        self.stacking_models()

    def get_path_transform(self, classifier_model):
        if self.stage is None:
            return os.path.join(self.path_transform, self.year, classifier_model)
        else:
            return os.path.join(self.path_transform, self.year, self.stage, classifier_model)

    def get_path_save_model(self):
        return os.path.join(self.path_model)

    def get_evaluate(self):
        return self.evaluate

    def get_data(self):
        return self.data

    def get_data_features(self):
        return self.new_data

    def load_all_data(self, schema):
        dic_data = {
            self.classifier_available[0]:  (self.spark.read.csv(self.get_path_transform(self.classifier_available[0]),
                                                                sep=",", header=True,
                                                                schema=schema)
                                            .withColumnRenamed("prediction", self.classifier_available[0]))
        }

        for classifier in self.classifier_available[1:]:
            dic_data[classifier] = (self.spark.read.csv(self.get_path_transform(classifier), sep=",", header=True,
                                                        schema=schema)
                                    .drop("label")
                                    .withColumnRenamed("prediction", classifier))
        return dic_data

    def merge_all_data(self, schema=get_data_schema("prediction"), id_column="id"):
        dic_data = self.load_all_data(schema)
        keys = dic_data.keys()
        self.data = dic_data[keys[0]]
        for key in keys[1:]:
            self.data = self.data.join(dic_data[key], on=id_column)
        self.data.count()

    def create_features(self):
        if len(self.classifier_available) == 2:
            udf_features = udf(lambda x, y: Vectors.dense([x, y]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(col(self.classifier_models[0]),
                                                                          col(self.classifier_models[1])))
        elif len(self.classifier_available) == 3:
            udf_features = udf(lambda x, y, z: Vectors.dense([x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(col(self.classifier_available[0]),
                                                                          col(self.classifier_available[1]),
                                                                          col(self.classifier_available[2])))
        elif len(self.classifier_available) == 4:
            udf_features = udf(lambda t, x, y, z: Vectors.dense([t, x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(col(self.classifier_available[0]),
                                                                          col(self.classifier_available[1]),
                                                                          col(self.classifier_available[2]),
                                                                          col(self.classifier_available[3])))
        elif len(self.classifier_available) == 5:
            udf_features = udf(lambda s, t, x, y, z: Vectors.dense([s, t, x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(col(self.classifier_available[0]),
                                                                          col(self.classifier_available[1]),
                                                                          col(self.classifier_available[2]),
                                                                          col(self.classifier_available[3]),
                                                                          col(self.classifier_available[4])))

        for classifier in self.classifier_available:
            self.new_data = self.new_data.drop(classifier)
        self.new_data.drop("matches").drop("team_1").drop("team_2")

    def stacking_models(self):
        classifier_model = ClassificationModel(self.spark, self.year, self.classifier_stacking, None,
                                               self.get_path_save_model(), None,
                                               validator="train_validation",
                                               list_layers=[[5, 3], [5, 4, 3], [5, 7, 7, 5, 7, 5, 3]])
        classifier_model.set_data(self.new_data)
        classifier_model.define_estimator()
        classifier_model.define_evaluator()
        classifier_model.define_grid_builder()
        classifier_model.define_validator()
        classifier_model.fit_validator()
        classifier_model.save_best_model()
        classifier_model.transform_model()
        self.evaluate = classifier_model.evaluate_evaluator()
        self.stacking_model = classifier_model.get_best_model()

    def stacking_transform(self):
        return self.stacking_model.transform(self.new_data)


if __name__ == "__main__":
    spark = get_spark_session("Stacking")
    # years = ["2014", "2010", "2006"]
    years = ["2018"]
    # Stacking classification models
    classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron",
                            "one_vs_rest"]
    dic_evaluate = {}
    for year in years:
        stacking = Stacking(spark, year, None, classification_model, "random_forest", "train_validation",
                            "./test/transform", "./test/stacking_model")
        stacking.run()
        dic_evaluate[year] = stacking.get_evaluate()

        stacking.stacking_transform().show(5)

    for key, value in dic_evaluate.iteritems():
        print("{0}: {1}".format(key, value))
