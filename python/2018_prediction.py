# -*- coding: utf-8 -*-
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

    matches = {"A": ["Russia-Saudi Arabia", "Egypt-Uruguay", "Russia-Egypt", "Uruguay-Saudi Arabia", "Uruguay-Russia",
                     "Saudi Arabia-Egypt"],
               "B": ["Morocco-Iran", "Portugal-Spain", "Portugal-Morocco", "Iran-Spain", "Iran-Portugal",
                     "Spain-Morocco"],
               "C": ["France-Australia", "Peru-Denmark", "Denmark-Australia", "France-Peru", "Denmark-France",
                     "Australia-Peru"],
               "D": ["Argentina-Iceland", "Croatia-Nigeria", "Argentina-Croatia", "Nigeria-Iceland", "Nigeria-Argentina",
                     "Iceland-Croatia"],
               "E": ["Costa Rica-Serbia", "Brazil-Switzerland", "Brazil-Costa Rica", "Serbia-Switzerland",
                     "Serbia-Brazil", "Switzerland-Costa Rica"],
               "F": ["Germany-Mexico", "Sweden-South Korea", "Germany-Sweden", "South Korea-Mexico",
                     "South Korea-Germany", "Mexico-Sweden"],
               "G": ["Belgium-Panama", "Tunisia-England", "Belgium-Tunisia", "England-Panama", "England-Belgium",
                     "Panama-Tunisia"],
               "H": ["Poland-Senegal", "Colombia-Japan", "Poland-Colombia", "Japan-Senegal", "Japan-Poland",
                     "Senegal-Colombia"]
               }

    def __init__(self, spark_session, year, group, model_classifier, path_model):
        self.spark = spark_session
        self.year = year
        self.group = group
        self.model_classifier = model_classifier
        self.path_model = path_model

        self.teams = None
        self.list_matches = []
        self.start = None
        self.data = None
        self.prediction = None

    def __str__(self):
        pass

    def run(self):
        self.load_data_start()
        self.load_teams()
        self.define_matches_features()
        self.transform_model()
        self.score_prediction()

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

    def append_matches(self):
        for matche in self.matches[self.group]:
            team_1 = self.teams.filter(col("country") == matche.split("-")[0]).rdd.map(lambda x: x["team"]).collect()[0]
            team_2 = self.teams.filter(col("country") == matche.split("-")[1]).rdd.map(lambda x: x["team"]).collect()[0]
            # print("{0}-{1}: {2}-{3}".format(matche.split("-")[0], matche.split("-")[1], team_1, team_2))
            self.list_matches.append((team_1, team_2))

    def define_matches_features(self):
        udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())
        udf_team1_team2 = udf(lambda team_1, team_2: team_1 + "/" + team_2, StringType())
        schema = StructType([
            StructField("team_1", StringType(), True),
            StructField("team_2", StringType(), True)])
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

    def transform_model(self):
        classification_model = ClassificationModel(None, self.year, self.model_classifier, None, self.path_model, None)
        self.prediction = (classification_model
                           .get_best_model()
                           .transform(self.data)
                           .select("team_1", "team_2", "prediction"))

    def score_prediction(self):
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

        team_points = (self.prediction
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

        team_points.select("country_1", "country_2", "prediction_1").show()

        team_1 = (team_points
                  .select("country_1", "points_1")
                  .rdd
                  .map(lambda x: (x["country_1"], x["points_1"]))
                  .reduceByKey(lambda x, y: x + y))

        team_2 = (team_points
                  .select("country_2", "points_2")
                  .rdd
                  .map(lambda x: (x["country_2"], x["points_2"]))
                  .reduceByKey(lambda x, y: x + y))

        team_union = team_1.union(team_2).reduceByKey(lambda x, y: x + y)
        print("{0}: {1}".format(group, sorted(team_union.collect(), key=lambda tp: tp[1], reverse=True)))


if __name__ == "__main__":
    spark = get_spark_session("2018 World Cup")
    for group in ["A"]:
        print("Group: {0}".format(group))
        world_cup = WorldCup(spark, "2018", group, "logistic_regression", "./test/classification_model")
        world_cup.run()

