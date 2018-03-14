# -*- coding: utf-8 -*-
import os
import numpy as np
from get_data_schema import get_data_schema
from get_spark_session import get_spark_session
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from result_statistic import ResultStatistic
from classification_model import ClassificationModel

class WorldCup:

    matches = {"1st_stage": {}}
    matches["1st_stage"]["A"] = ["Russia-Saudi Arabia", "Egypt-Uruguay", "Russia-Egypt", "Uruguay-Saudi Arabia",
                                 "Uruguay-Russia", "Saudi Arabia-Egypt"]
    matches["1st_stage"]["B"] = ["Morocco-Iran", "Portugal-Spain", "Portugal-Morocco", "Iran-Spain", "Iran-Portugal",
                                 "Spain-Morocco"]
    matches["1st_stage"]["C"] = ["France-Australia", "Peru-Denmark", "Denmark-Australia", "France-Peru",
                                 "Denmark-France", "Australia-Peru"]
    matches["1st_stage"]["D"] = ["Argentina-Iceland", "Croatia-Nigeria", "Argentina-Croatia", "Nigeria-Iceland",
                                 "Nigeria-Argentina", "Iceland-Croatia"]
    matches["1st_stage"]["E"] = ["Costa Rica-Serbia", "Brazil-Switzerland", "Brazil-Costa Rica", "Serbia-Switzerland",
                                 "Serbia-Brazil", "Switzerland-Costa Rica"]
    matches["1st_stage"]["F"] = ["Germany-Mexico", "Sweden-South Korea", "Germany-Sweden", "South Korea-Mexico",
                                 "South Korea-Germany", "Mexico-Sweden"]
    matches["1st_stage"]["G"] = ["Belgium-Panama", "Tunisia-England", "Belgium-Tunisia", "England-Panama",
                                 "England-Belgium", "Panama-Tunisia"]
    matches["1st_stage"]["H"] = ["Poland-Senegal", "Colombia-Japan", "Poland-Colombia", "Japan-Senegal", "Japan-Poland",
                                 "Senegal-Colombia"]

    classification_models = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron",
                             "one_vs_rest"]

    def __init__(self, spark_session, year, stage, model_classifier, stacking, path_model, path_prediction):
        self.spark = spark_session
        self.year = year
        self.stage = stage
        self.model_classifier = model_classifier
        self.stacking = stacking
        self.path_model = path_model
        self.path_prediction = path_prediction

        self.teams = None
        self.list_matches = []
        self.start = None
        self.data = None
        self.prediction = None

    def __str__(self):
        pass

    def run(self):
        if not self.stacking:
            self.load_data_start()
            self.define_matches_features()
        else:
            self.define_prediction_features()
        self.transform_model()
        self.save_prediction()
        self.score_prediction()

    def get_path_prediction(self, stacking, classifier):
        if stacking:
            return os.path.join(self.path_prediction, self.year, self.stage, "stacking", classifier)
        else:
            return os.path.join(self.path_prediction, self.year, self.stage, classifier)

    def load_data_start(self):
        udf_get_percentage_game = udf(lambda x, y: x / y, FloatType())

        def convert_string_to_float(x):
            x_replace_minus = x.replace(u'\u2212', '-')
            if x_replace_minus == '-':
                return np.nan
            else:
                return float(x_replace_minus)

        udf_convert_string_to_float = udf(lambda x: convert_string_to_float(x), FloatType())
        udf_create_features = udf(lambda s, t, u, v, w, x, y, z: Vectors.dense([s, t, u, v, w, x, y, z]), VectorUDT())

        names_start_to_convert = get_data_schema("qualifying_start").names
        names_start_to_convert.remove("teamGroup_team")
        path = "./data/2018_World_Cup.tsv"
        self.start = (self.spark.read.csv(path, sep="\t", schema=get_data_schema("qualifying_start"), header=False)
                .select([udf_convert_string_to_float(col(name)).alias(name) for name in names_start_to_convert] +
                        ["teamGroup_team"])
                .withColumn("features", udf_create_features(
            udf_get_percentage_game(col("matchesGroup_home"), col("matchesGroup_total")),
            udf_get_percentage_game(col("matchesGroup_away"), col("matchesGroup_total")),
            udf_get_percentage_game(col("matchesGroup_neutral"), col("matchesGroup_total")),
            udf_get_percentage_game(col("matchesGroup_wins"), col("matchesGroup_total")),
            udf_get_percentage_game(col("matchesGroup_losses"), col("matchesGroup_total")),
            udf_get_percentage_game(col("matchesGroup_draws"), col("matchesGroup_total")),
            udf_get_percentage_game(col("goalsGroup_for"), col("matchesGroup_total")),
            udf_get_percentage_game(col("goalsGroup_against"), col("matchesGroup_total"))))
                .withColumnRenamed("teamGroup_team", "team")
                .select("team", "features"))

    def load_teams(self):
        self.teams = ResultStatistic(self.spark, None, None, None, None).load_data_teams()

    def save_prediction(self):
        (self.prediction
         .write
         .csv(self.get_path_prediction(self.stacking, self.model_classifier), mode="overwrite", sep=",", header=True))

    def append_matches(self):
        for group in self.matches[self.stage].keys():
            for match in self.matches[self.stage][group]:
                team_1 = (self.teams.filter(col("country") == match.split("-")[0])
                                    .rdd.map(lambda x: x["team"]).collect()[0])
                team_2 = (self.teams.filter(col("country") == match.split("-")[1])
                              .rdd.map(lambda x: x["team"]).collect()[0])
                # print("{0}-{1}: {2}-{3}".format(matche.split("-")[0], matche.split("-")[1], team_1, team_2))
                self.list_matches.append((group, team_1, team_2))

    def define_matches_features(self):
        udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())
        udf_team1_team2 = udf(lambda team_1, team_2: team_1 + "/" + team_2, StringType())
        schema = StructType([
            StructField("group", StringType(), False),
            StructField("team_1", StringType(), False),
            StructField("team_2", StringType(), False)])
        self.append_matches()
        self.data = (self.spark.createDataFrame(self.list_matches, schema=schema)
                      .join(self.start, col("team_1") == col("team"))
                      .drop("team")
                      .withColumnRenamed("features", "features_1")
                      .join(self.start, col("team_2") == col("team"))
                      .drop("team")
                      .withColumnRenamed("features", "features_2")
                      .withColumn("features", udf_diff_features(col("features_1"), col("features_2")))
                      .withColumn("matches", udf_team1_team2(col("team_1"), col("team_2"))))

    def define_prediction_features(self):
        schema = StructType([
            StructField("team_1", StringType(), True),
            StructField("team_2", StringType(), True),
            StructField("matches", StringType(), True),
            StructField("prediction", FloatType(), True)
        ])
        dic_data = {}
        for classifier in self.classification_models:
            dic_data[classifier] = (self.spark.read.csv(self.get_path_prediction(False, classifier), sep=",",
                                                        schema=schema, header=True)
                                    .drop("team_1").drop("team_2")
                                    .withColumnRenamed("prediction", classifier))
        keys = dic_data.keys()
        self.data = dic_data[keys[0]]
        for key in keys[1:]:
            self.data = self.data.join(dic_data[key], on="matches")
        self.data.count()

        if len(self.classification_models) == 2:
            udf_features = udf(lambda x, y: Vectors.dense([x, y]), VectorUDT())
            self.data = self.data.withColumn("features", udf_features(col(self.classification_models[0]),
                                                                      col(self.classification_models[1])))
        elif len(self.classification_models) == 3:
            udf_features = udf(lambda x, y, z: Vectors.dense([x, y, z]), VectorUDT())
            self.data = self.data.withColumn("features", udf_features(col(self.classification_models[0]),
                                                                      col(self.classification_models[1]),
                                                                      col(self.classification_models[2])))
        elif len(self.classification_models) == 4:
            udf_features = udf(lambda t, x, y, z: Vectors.dense([t, x, y, z]), VectorUDT())
            self.data = self.data.withColumn("features", udf_features(col(self.classification_models[0]),
                                                                      col(self.classification_models[1]),
                                                                      col(self.classification_models[2]),
                                                                      col(self.classification_models[3])))
        elif len(self.classification_models) == 5:
            udf_features = udf(lambda s, t, x, y, z: Vectors.dense([s, t, x, y, z]), VectorUDT())
            self.data = self.data.withColumn("features", udf_features(col(self.classification_models[0]),
                                                                      col(self.classification_models[1]),
                                                                      col(self.classification_models[2]),
                                                                      col(self.classification_models[3]),
                                                                      col(self.classification_models[4])))
        self.data.select("matches", "features")

    def transform_model(self):
        classification_model = ClassificationModel(None, self.year, self.model_classifier, None, self.path_model, None)
        self.prediction = (classification_model
                           .get_best_model()
                           .transform(self.data)
                           .select("matches", "prediction"))

    def score_prediction(self):
        self.load_teams()
        def prediction_team2(x):
            if x == 2.0:
                return 1.0
            elif x == 1.0:
                return 2.0
            else:
                return 0.0
        udf_prediction_team2 = udf(lambda x: prediction_team2(x), FloatType())

        def score_point(x):
            if x == 2.0:
                return 3
            elif x == 1.0:
                return 0
            elif x == 0.0:
                return 1
        udf_points = udf(lambda x: score_point(x), IntegerType())

        udf_team_1 = udf(lambda x: x.split("/")[0], StringType())
        udf_team_2 = udf(lambda x: x.split("/")[1], StringType())

        team_points = (self.prediction
         .withColumn("team_1", udf_team_1(col("matches")))
         .withColumn("team_2", udf_team_2(col("matches")))
         .join(self.teams, col("team_1") == col("team"))
         .drop("team")
         .withColumnRenamed("country", "country_1")
         .join(self.teams, col("team_2") == col("team"))
         .drop("team")
         .withColumnRenamed("country", "country_2")
         .withColumnRenamed("prediction", "prediction_1")
         .withColumn("prediction_2", udf_prediction_team2(col("prediction_1")))
         .withColumn("points_1", udf_points(col("prediction_1")))
         .withColumn("points_2", udf_points(col("prediction_2"))))

        # team_points.show()
        team_points.select("country_1", "country_2", "points_1").show()


if __name__ == "__main__":
    spark = get_spark_session("2018 World Cup")
    # classification_models = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron",
    #                          "one_vs_rest"]
    # for classifier in classification_models:
    #     print("Classifer: {0}".format(classifier))
    #     world_cup = WorldCup(spark, "2018", "1st_stage", classifier, False,
    #                          "./test/classification_model", "./test/prediction")
    #     world_cup.run()

    world_cup = WorldCup(spark, "2018", "1st_stage", "random_forest", True,
                             "./test/stacking_model", "./test/prediction")
    world_cup.run()
