
import os

from pyspark.ml.feature import IndexToString, StringIndexerModel
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType

from classification_model import ClassificationModel
from regression_model import RegressionModel
from featurization_data import FeaturizationData
from stacking_classification_method import Stacking
from get_spark_session import get_spark_session
from get_competition_dates import get_competition_dates


class CompleteStage:

    schema = StructType([
        StructField("matches", StringType(), True),
        StructField("label", FloatType(), True),
        StructField("prediction", FloatType(), True)])

    def __init__(self, spark_session, year, model, model_method, stage, list_date, path_data, path_model,
                 path_prediction):
        self.spark = spark_session
        self.year = year
        self.model = model
        self.model_method = model_method  # "classification", "regression", "stacking"
        self.stage = stage
        self.list_date = list_date
        self.path_data = path_data
        self.path_model = path_model
        self.path_prediction = path_prediction

        self.label_prediction = None
        self.transform = None
        self.evaluate = None
    
    def __str__(self):
        s = "FirstStage\n"
        s += "Spark session: {0}\n".format(self.spark)
        s += "Year: {0}".format(self.year)
        s += "Model classifier: {0}\n".format(self.model_classifier)
        s += "Stacking model: {0}\n".format(self.stacking_test)
        s += "List dates: {0}\n".format(self.list_date)
        s += "Path data: {0}\n".format(self.path_data)
        s += "Path model: {0}\n".format(self.path_model)
        s += "Path prediction: {0}\n".format(self.path_prediction)
        return s
    
    def run(self):
        data = self.load_data_stage()
        if self.model_method in ["classification", "regression"]:
            self.transform_model(data)
            self.save_prediction()
        else:
            available_classifier = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron",
                                    "one_vs_rest"]
            stacking = Stacking(self.spark, self.year, self.stage, available_classifier, self.model_classifier, "train_validation",
                                self.path_data, self.path_model)
            stacking.merge_all_data(schema=self.schema, id_column="matches")
            stacking.create_features()
            # stacking.get_data().show(10)
            # stacking.get_data_features().show(10)
            self.transform_model(stacking.get_data_features())
            self.save_prediction()

    def load_data_stage(self):
        featurization_data = FeaturizationData(self.spark, self.year, ["WCF"], None, None,
                                               stage=self.stage, list_date=self.list_date)
        featurization_data.get_dates()
        featurization_data.loop_all_confederations()
        featurization_data.union_all_confederation()
        return featurization_data.get_data_union()

    def load_data_teams(self):
        schema = StructType([StructField("team", StringType(), True),
                             StructField("country", StringType(), True)])
        return self.spark.read.csv("../data/common/en.teams.tsv", sep="\t", header=False, schema=schema)

    def get_data(self):
        return self.data

    def get_transform(self):
        return self.transform

    def get_evaluate(self):
        return self.evaluate

    def get_prediction_path(self):
        if self.model_method == "stacking":
            return os.path.join(self.path_prediction, self.year, self.stage, "stacking", self.model)
        else:
            return os.path.join(self.path_prediction, self.year, self.stage, self.model)

    def apply_index_to_string(self):
        teams = self.load_data_teams()
        labels = StringIndexerModel.load("../test/string_indexer").labels
        model = IndexToString(inputCol="id", outputCol="matches", labels=labels)

        udf_get_team_1 = udf(lambda x: x.split("_")[0].split("/")[0], StringType())
        udf_get_team_2 = udf(lambda x: x.split("_")[0].split("/")[1], StringType())
        udf_get_date = udf(lambda x: x.split("_")[1], StringType())         

        def result_team_2(result):
            if result == 2:
                return 1.0
            elif result == 1:
                return 2.0
            else:
                return 0.0

        udf_result_team_2 = udf(lambda result: result_team_2(result), FloatType())

        self.transform.show()
        model.transform(self.transform).show()
        self.data = (model.transform(self.transform)
                     .withColumn("team_1", udf_get_team_1(col("matches")))
                     .withColumn("team_2", udf_get_team_2(col("matches")))
                     .withColumn("date", udf_get_date(col("matches"))) 
                     .withColumn("result_team_2", udf_result_team_2(col("prediction")))
                     .withColumnRenamed("prediction", "result_team_1")
                     .select("date", "team_1", "team_2", "result_team_1", "result_team_2"))

        self.data = (self.data.join(teams, self.data.team_1 == teams.team)
                    .withColumnRenamed("country", "country_1").drop("team")
                    .join(teams, self.data.team_2 == teams.team)
                    .withColumnRenamed("country", "country_2").drop("team"))

    def join_label_prediction(self, label, prediction):
        udf_matches = udf(lambda team_1,team_2: team_1 + "/" + team_2, StringType())
        label = label.withColumn("matches", udf_matches(col("team_1"), col("team_2")))
        prediction = (prediction
                      .withColumn("matches", udf_matches(col("team_1"), col("team_2")))
                      .select("matches", "prediction"))
        self.label_prediction = label.join(prediction, on="matches").select("id", "label", "prediction")

    def define_matches_features(self, data):
        udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())
        team_features = self.get_team_features()
        return (data
                .join(team_features, col("team_1") == col("team"))
                .drop("team").withColumnRenamed("features", "features_1")
                .join(team_features, col("team_2") == col("team"))
                .drop("team").withColumnRenamed("features", "features_2")
                .withColumn("features", udf_diff_features(col("features_1"), col("features_2"))))
        
    def transform_model(self, data):
        if self.model_method == "classification":
            model = ClassificationModel(None, self.year, self.model, None, self.path_model, None, None)
        elif self.model_method == "regression":
            model = RegressionModel(None, self.year, self.model, None, self.path_model, None, None)
        model.load_best_model()
        model.set_data(data)
        model.transform_model()
        if self.model_method == "regression":
            model.define_match_issue()
        model.define_evaluator()
        self.transform = model.get_transform()

        evaluator = ClassificationModel(None, None, None, None, None, None, None)
        evaluator.define_evaluator()
        evaluator.set_transform(self.transform)
        self.evaluate = evaluator.evaluate_evaluator()

    def save_matches_next_stage(self, dic_first_by_group, dic_second_by_group):
        data = []
        for tp in self.tp_groups:
            data.append(('/'.join(dic_first_by_group[tp[0]]), '/'.join(dic_second_by_group[tp[1]]))) 
            data.append(('/'.join(dic_first_by_group[tp[1]]), '/'.join(dic_second_by_group[tp[0]])))
        (self.spark.createDataFrame(data, ["team_1", "team_2"])
         .write
         .csv("./test/matches_next_stage/{0}".format(self.model_classifier), mode="overwrite", sep=",", header=True))

    def save_prediction(self):
        (self.transform
         .select("matches", "label", "prediction")
         .coalesce(1)
         .write.csv(self.get_prediction_path(), mode="overwrite", sep=",", header=True))


if __name__ == "__main__":
    spark = get_spark_session("First Stage")
    model_methods = ["classification", "regression"]  # stacking

    years = ["2014", "2010", "2006"]
    classification_models = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron",
                            "one_vs_rest"]
    regression_models = ["linear_regression", "decision_tree", "random_forest", "gbt_regressor"]
    dic_model_methods = {method: None for method in model_methods}
    dic_model_methods["classification"] = {year: classification_models for year in years}
    dic_model_methods["regression"] = {year: regression_models for year in years}
    dic_path_models = {
        "classification": "./test/model/classification_model",
        "regression": "./test/model/regression_model"
    }
    dic_path_prediction = {
        "classification": "./test/prediction/classification_model",
        "regression": "./test/prediction/regression_model"
    }

    dic_evaluate = {}
    for method in dic_model_methods.keys():
        dic_evaluate[method] = {}
        print("Method: {0}".format(method))
        for year in dic_model_methods[method].keys():
            dic_evaluate[method][year] = {}
            print("  Year: {0}".format(year))
            for model in dic_model_methods[method][year]:
                dic_evaluate[method][year][model] = {}
                for stage in get_competition_dates(year).keys():
                    print("  Model: {0}".format(model))
                    complete_stage = CompleteStage(spark, year, model, method, stage, get_competition_dates(year)[stage],
                                                   None, dic_path_models[method], dic_path_prediction[method])
                    # print(first_stage)
                    complete_stage.run()
                    dic_evaluate[method][year][model][stage] = complete_stage.get_evaluate()



        # for classifier in ["logistic_regression", "decision_tree", "random_forest"]:
        #     for stage in get_competition_dates(year).keys():
        #         first_stage = FirstStage(spark, year, classifier, True, stage,
        #                                  get_competition_dates(year)[stage],
        #                                  "./test/prediction", "./test/stacking_model", "./test/prediction")
        #         first_stage.run()
