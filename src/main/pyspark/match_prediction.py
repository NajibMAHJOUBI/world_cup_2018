# -*- coding: utf-8 -*-
import os

import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from classification_model import ClassificationModel
from get_data_schema import get_data_schema
from regression_model import RegressionModel


class MatchPrediction:

    schema_match = StructType([StructField("team_1", StringType(), False),
                               StructField("team_2", StringType(), False)])

    def __init__(self, spark_session, year, path_model,
                 model_classification=None, model_regression=None, model_stacking=None, stacking_name=None):
        self.spark = spark_session
        self.year = year
        self.path_model = path_model
        self.model_classification = model_classification
        self.model_regression = model_regression
        self.model_stacking = model_stacking
        self.stacking_name = stacking_name

        self.start = None
        self.teams = None
        self.country_1, self.country_2 = None, None
        self.team_1, self.team_2 = None, None
        self.data = None
        self.prediction = None

        self.load_data_teams()
        self.load_data_start()

    def __str__(self):
        s = "{0} - {1}".format(self.country_1, self.country_2)
        return s

    def run(self):
        self.define_matches_features()
        self.transform_model()

    def get_teams(self):
        return self.team_1, self.team_2

    def get_country(self):
        return self.country_1, self.country_2

    def get_dic_teams(self):
        return self.teams

    def get_path_model(self):
        if self.model_classification is not None:
            return os.path.join(self.path_model, "classification")
        elif self.model_regression is not None:
            return os.path.join(self.path_model, "regression")

    def get_prediction(self):
        return self.prediction.rdd.map(lambda x: x["prediction"]).collect()[0]

    def set_teams(self):
        self.team_1 = str(self.teams[self.country_1])
        self.team_2 = str(self.teams[self.country_2])

    def set_country(self, country_1, country_2):
        self.country_1 = country_1
        self.country_2 = country_2
        self.set_teams()

    def set_model_regression(self, model):
        self.model_regression = model

    def set_model_classification(self, model):
        self.model_classification = model

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
        path = "./data/WCF/2018_World_Cup_WCP_qualifying_start.tsv"
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

    def load_data_teams(self):
        data = (self.spark.read.csv("./data/common/en.teams.tsv", sep="\t", header=False,
                                    schema=get_data_schema("teams"))
                .select("country", "team")
                .toPandas()
                .to_dict('list'))
        self.teams = {team: data["team"][index] for index,team in enumerate(data["country"])}

    def define_matches_features(self):
        udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())
        udf_team1_team2 = udf(lambda team_1, team_2: team_1 + "/" + team_2, StringType())

        rdd = self.spark.sparkContext.parallelize([(self.team_1, self.team_2)])
        self.data = (self.spark.createDataFrame(rdd, schema=self.schema_match)
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
        if self.model_classification is not None:
            self.model = ClassificationModel(None, self.year, self.model_classification, None, self.get_path_model(),
                                              None, None)
        elif self.model_regression is not None:
            self.model = RegressionModel(None, self.year, self.model_regression, None, self.get_path_model(),
                                         None, None)

        self.prediction = (self.model
                           .get_best_model()
                           .transform(self.data))

        if self.model_regression is not None:
            self.model.set_transform(self.prediction)
            self.model.define_match_issue()
            self.prediction = self.model.get_transform()

        self.prediction = self.prediction.select("matches", "prediction")

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
    from get_spark_session import get_spark_session
    from get_world_cup_matches import get_matches
    from get_classification_approach import get_classification_approach
    from get_regression_approach import get_regression_approach
    year = "2018"
    spark = get_spark_session("2018 World Cup")
    matches = get_matches(year)

    # world_cup = MatchPrediction(spark, year, "./test/pyspark/model",
    #                             model_classification="decision_tree",
    #                             model_regression=None,
    #                             model_stacking=None,
    #                             stacking_name=None)
    #
    # for group in sorted(matches["1st_stage"].keys()):
    #     print("Group: {0}".format(group))
    #     for match in matches["1st_stage"][group]:
    #         country_1, country_2 = match
    #         world_cup.set_country(country_1, country_2)
    #         world_cup.run()
    #         print("  {0}: {1}".format(world_cup.get_country(), world_cup.get_prediction()))
    #     print("")

    models = {"classification": get_classification_approach(),
              "regression": get_regression_approach()}

    world_cup = MatchPrediction(spark, year, "./test/pyspark/model",
                                model_classification=None,
                                model_regression=None,
                                model_stacking=None,
                                stacking_name=None)
    world_cup.set_country("Morocco", "Portugal")
    print("{0}".format(world_cup.get_country()))

    for model_type, models_ in models.iteritems():
        for model in models_:
            if model_type == "classification":
                world_cup.set_model_classification(model)
                world_cup.set_model_regression(None)
            elif model_type == "regression":
                world_cup.set_model_regression(model)
                world_cup.set_model_classification(None)
            world_cup.run()
            print("{0}; {1}:  {2}".format(model_type, model, world_cup.get_prediction()))

