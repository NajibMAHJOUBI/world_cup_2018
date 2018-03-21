import os

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf

from classification_model import ClassificationModel
from get_data_schema import get_data_schema
from get_spark_session import get_spark_session
from get_stacking_approach import get_stacking_approach


class StackingModels:
    def __init__(self, spark_session, year, stacking, classifiers, validator, path_transform, path_model, model_name):
        self.spark = spark_session
        self.year = year
        self.stacking = stacking
        self.classifiers = classifiers
        self.validator = validator
        self.path_transform = path_transform  # "./test/transform"
        self.path_model = path_model
        self.model_name = model_name

        self.data = None
        self.new_data = None
        self.evaluate = None
        self.stacking_model = None

    def __str__(self):
        s = "Stacking model\n"
        s += "Stacking classifier: {0}".format(self.classifiers)
        return s

    def run(self):
        self.merge_all_data()
        self.create_features()
        self.stacking_models()

    def get_path_transform(self, method, model):
        return os.path.join(self.path_transform, method, self.year, model)

    def get_path_save_model(self):
        return os.path.join(self.path_model, "stacking", self.model_name)

    def get_evaluate(self):
        return self.evaluate

    def get_data(self):
        return self.data

    def get_data_features(self):
        return self.new_data

    def load_all_data(self, schema):
        dic_data = {}
        features_name = '_'.join([self.classifiers[0][0], self.classifiers[0][1]])
        dic_data[features_name] = (self.spark.read.csv(self.get_path_transform(
            self.classifiers[0][0], self.classifiers[0][1]), sep=",", header=True, schema=schema)
                                   .withColumnRenamed("prediction", features_name))

        for method, model in self.classifiers[1:]:
            features_name = '_'.join([method, model])
            dic_data[features_name] = (self.spark.read.csv(self.get_path_transform(method, model), sep=",", header=True,
                                                           schema=schema)
                                       .drop("label")
                                       .withColumnRenamed("prediction", features_name))
        return dic_data

    def merge_all_data(self, schema=get_data_schema("prediction"), id_column="id"):
        dic_data = self.load_all_data(schema)
        keys = dic_data.keys()
        self.data = dic_data[keys[0]]
        for key in keys[1:]:
            self.data = self.data.join(dic_data[key], on=id_column)
        self.data.count()

    def create_features(self):
        if len(self.classifiers) == 2:
            udf_features = udf(lambda x, y: Vectors.dense([x, y]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(
                                                       col('_'.join([self.classifiers[0][0], self.classifiers[0][1]])),
                                                       col('_'.join([self.classifiers[1][0], self.classifiers[1][1]]))))
        elif len(self.classifiers) == 3:
            udf_features = udf(lambda x, y, z: Vectors.dense([x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(
                                                    col('_'.join([self.classifiers[0][0], self.classifiers[0][1]])),
                                                    col('_'.join([self.classifiers[1][0], self.classifiers[1][1]])),
                                                    col('_'.join([self.classifiers[2][0], self.classifiers[2][1]]))))
        elif len(self.classifiers) == 4:
            udf_features = udf(lambda t, x, y, z: Vectors.dense([t, x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(
                                                       col('_'.join([self.classifiers[0][0], self.classifiers[0][1]])),
                                                       col('_'.join([self.classifiers[1][0], self.classifiers[1][1]])),
                                                       col('_'.join([self.classifiers[2][0], self.classifiers[2][1]])),
                                                       col('_'.join([self.classifiers[3][0], self.classifiers[3][1]]))))
        elif len(self.classifiers) == 5:
            udf_features = udf(lambda s, t, x, y, z: Vectors.dense([s, t, x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(
                                                       col('_'.join([self.classifiers[0][0], self.classifiers[0][1]])),
                                                       col('_'.join([self.classifiers[1][0], self.classifiers[1][1]])),
                                                       col('_'.join([self.classifiers[2][0], self.classifiers[2][1]])),
                                                       col('_'.join([self.classifiers[3][0], self.classifiers[3][1]])),
                                                       col('_'.join([self.classifiers[4][0], self.classifiers[4][1]]))))
        elif len(self.classifiers) == 6:
            udf_features = udf(lambda r, s, t, x, y, z: Vectors.dense([r, s, t, x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(
                                                       col('_'.join([self.classifiers[0][0], self.classifiers[0][1]])),
                                                       col('_'.join([self.classifiers[1][0], self.classifiers[1][1]])),
                                                       col('_'.join([self.classifiers[2][0], self.classifiers[2][1]])),
                                                       col('_'.join([self.classifiers[3][0], self.classifiers[3][1]])),
                                                       col('_'.join([self.classifiers[4][0], self.classifiers[4][1]])),
                                                       col('_'.join([self.classifiers[5][0], self.classifiers[5][1]]))))
        elif len(self.classifiers) == 7:
            udf_features = udf(lambda q, r, s, t, x, y, z: Vectors.dense([q, r, s, t, x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(
                                                       col('_'.join([self.classifiers[0][0], self.classifiers[0][1]])),
                                                       col('_'.join([self.classifiers[1][0], self.classifiers[1][1]])),
                                                       col('_'.join([self.classifiers[2][0], self.classifiers[2][1]])),
                                                       col('_'.join([self.classifiers[3][0], self.classifiers[3][1]])),
                                                       col('_'.join([self.classifiers[4][0], self.classifiers[4][1]])),
                                                       col('_'.join([self.classifiers[5][0], self.classifiers[5][1]])),
                                                       col('_'.join([self.classifiers[6][0], self.classifiers[6][1]]))))
        elif len(self.classifiers) == 8:
            udf_features = udf(lambda p, q, r, s, t, x, y, z: Vectors.dense([p, q, r, s, t, x, y, z]), VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(
                                                       col('_'.join([self.classifiers[0][0], self.classifiers[0][1]])),
                                                       col('_'.join([self.classifiers[1][0], self.classifiers[1][1]])),
                                                       col('_'.join([self.classifiers[2][0], self.classifiers[2][1]])),
                                                       col('_'.join([self.classifiers[3][0], self.classifiers[3][1]])),
                                                       col('_'.join([self.classifiers[4][0], self.classifiers[4][1]])),
                                                       col('_'.join([self.classifiers[5][0], self.classifiers[5][1]])),
                                                       col('_'.join([self.classifiers[6][0], self.classifiers[6][1]])),
                                                       col('_'.join([self.classifiers[7][0], self.classifiers[7][1]]))))
        elif len(self.classifiers) == 9:
            udf_features = udf(lambda o, p, q, r, s, t, x, y, z: Vectors.dense([o, p, q, r, s, t, x, y, z]),
                               VectorUDT())
            self.new_data = self.data.withColumn("features", udf_features(
                                                       col('_'.join([self.classifiers[0][0], self.classifiers[0][1]])),
                                                       col('_'.join([self.classifiers[1][0], self.classifiers[1][1]])),
                                                       col('_'.join([self.classifiers[2][0], self.classifiers[2][1]])),
                                                       col('_'.join([self.classifiers[3][0], self.classifiers[3][1]])),
                                                       col('_'.join([self.classifiers[4][0], self.classifiers[4][1]])),
                                                       col('_'.join([self.classifiers[5][0], self.classifiers[5][1]])),
                                                       col('_'.join([self.classifiers[6][0], self.classifiers[6][1]])),
                                                       col('_'.join([self.classifiers[7][0], self.classifiers[7][1]])),
                                                       col('_'.join([self.classifiers[8][0], self.classifiers[8][1]]))))

        for method, model in self.classifiers:
            self.new_data = self.new_data.drop('_'.join([method, model]))
        self.new_data.drop("matches").drop("team_1").drop("team_2")

    def stacking_models(self):
        classifier_model = ClassificationModel(self.spark, self.year, self.stacking, None,
                                               self.get_path_save_model(), None, "train_validation",)
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
    model_stacking = "logistic_regression"
    if not os.path.isdir("./test/accuracy/{0}".format(model_stacking)):
        os.system("mkdir ./test/accuracy/{0}".format(model_stacking))
    years = ["2014"]
    stacking_approaches = get_stacking_approach()
    # dic_evaluate = {}
    for year in years:
        number_classifiers = 5
        f = open("./test/accuracy/{0}/{1}_{2}_stacking_accuracy.csv".format(model_stacking, year, number_classifiers),
                 "a+")
        for index, value in enumerate(filter(lambda x: len(x[1]) == number_classifiers, stacking_approaches.items())):
            model_name, classifiers = value[0], value[1]
            if not os.path.isdir("./test/model/stacking/{0}/{1}/{2}".format(model_name, year, model_stacking)):
                print(index, model_name)
                # dic_evaluate[model_name] = {}
                stacking_model = StackingModels(spark, year, model_stacking, classifiers, "train_validation",
                                                "./test/transform", "./test/model", model_name)
                stacking_model.run()
                # dic_evaluate[model_name][year] = stacking_model.get_evaluate()
                f.write("{0};{1}\n".format(classifiers, stacking_model.get_evaluate()))
        f.close()

    # for key, value in dic_evaluate.iteritems():
    #     print("{0}: {1}".format(key, value))
