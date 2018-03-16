# -*- coding: utf-8 -*-
import os

from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType, IntegerType
from pyspark.sql.functions import col, udf

from get_spark_session import get_spark_session
from get_competition_dates import get_competition_dates
from classification_model import ClassificationModel


def team_result(results):
    resu = 0.0
    for tp in results:
        resu += tp[0] * tp[1]
    return resu


class ResultStatistic:

    schema = StructType([StructField("matches", StringType(), True),
                         StructField("label", DoubleType(), True),
                         StructField("prediction", DoubleType(), True)])
    schema_groups = StructType([StructField("group", StringType(), True),
                                StructField("country_1", StringType(), True),
                                StructField("country_2", StringType(), True),
                                StructField("country_3", StringType(), True),
                                StructField("country_4", StringType(), True)])
    schema_teams = StructType([StructField("team", StringType(), True),
                               StructField("country", StringType(), True)])
    stages = ["1st_stage", "2nd_stage", "3rd_stage", "4th_stage", "5th_stage", "6th_stage"]

    def __init__(self, spark_session, year, classifier_model, stacking, path_data, stage=None):
        self.spark = spark_session
        self.year = year
        self.classifier_model = classifier_model
        self.stacking = stacking
        self.path_data = path_data
        self.stage = stage

        self.vs_groups = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]

        self.label_prediction = None

    def __str__(self):
        s = "Result Statistic\n"
        s += "Spark session: {0}\n".format(self.spark)
        s += "Year: {0}\n".format(self.year)
        s += "Stage: {0}\n".format(self.stage)
        return s

    def run(self):
        self.load_label_prediction()

    def set_stage(self, stage_to_set):
        self.stage = stage_to_set

    def get_path_label_prediction(self):
        if self.stacking:
            return os.path.join(self.path_data, self.year, self.stage, "stacking", self.classifier_model)
        else:
            return os.path.join(self.path_data, self.year, self.stage, self.classifier_model)

    def load_label_prediction(self):
        self.label_prediction = self.spark.read.csv(self.get_path_label_prediction(), header=True, sep=",",
                                                    schema=self.schema)

    def load_data_teams(self):
        return self.spark.read.csv("./data/common/en.teams.tsv", sep="\t", header=False, schema=self.schema_teams)

    def load_data_groups(self):
        teams = self.load_data_teams()
        data = (self.spark.read.csv("./data/groups/{0}_groups.csv".format(self.year), sep=",",
                                    schema=self.schema_groups, header=False)
                .rdd
                .map(lambda x: (x["group"], x["country_1"], x["country_2"], x["country_3"], x["country_4"]))
                .map(lambda tp: [(tp[0], item) for item in tp[1:]]).reduce(lambda x, y: x + y))
        return self.spark.createDataFrame(data, ["group", "country"]).join(teams, on="country")

    def merge_label_prediction(self):
        self.label_prediction = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD(), self.schema)
        for stage in self.stages:
            self.set_stage(stage)
            data = self.spark.read.csv(self.get_path_label_prediction(), header=True, sep=",", schema=self.schema)
            self.label_prediction = self.label_prediction.union(data)
        self.label_prediction.count()

    def compute_accuracy(self):
        classification_model = ClassificationModel(None, None, None, None, None, None)
        classification_model.set_transform(self.label_prediction)
        classification_model.define_evaluator()
        return classification_model.evaluate_evaluator()

    def compute_accuracy_by_stage(self):
        self.load_label_prediction()
        return self.compute_accuracy()

    def compute_accuracy_global(self):
        self.merge_label_prediction()
        return self.compute_accuracy()

    def scores_first_stage(self):
        self.set_stage("1st_stage")
        self.load_label_prediction()
        groups = self.load_data_groups()

        udf_team1 = udf(lambda x: x.split("_")[0].split("/")[0], StringType())
        udf_team2 = udf(lambda x: x.split("_")[0].split("/")[1], StringType())

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

        return (self.label_prediction
                 .withColumn("team_1", udf_team1(col("matches")))
                 .withColumn("team_2", udf_team2(col("matches")))
                 .withColumn("prediction_2", udf_prediction_team2(col("prediction")))
                 .withColumnRenamed("prediction", "prediction_1")
                 .withColumn("points_1", udf_points(col("prediction_1")))
                 .withColumn("points_2", udf_points(col("prediction_2")))
                 .join(groups, col("team_1") == col("team")).drop("group").drop("team")
                 .withColumnRenamed("country", "country_1")
                 .join(groups, col("team_2") == col("team")).drop("team")
                 .withColumnRenamed("country", "country_2")
                 .select("country_1", "points_1", "country_2", "points_2", "group"))

    def print_scores_by_group(self):
        points = self.scores_first_stage()
        groups = points.select("group").distinct().rdd.map(lambda x: x["group"]).collect()

        rdd_team_1 = (points
                      .rdd
                      .map(lambda x: ((x["country_1"], x["group"]), x["points_1"]))
                      .reduceByKey(lambda x, y: x + y))
        rdd_team_2 = (points
                      .rdd
                      .map(lambda x: ((x["country_2"], x["group"]), x["points_2"]))
                      .reduceByKey(lambda x, y: x + y))
        rdd_points = (rdd_team_1
                      .union(rdd_team_2)
                      .reduceByKey(lambda x, y: x + y)
                      .map(lambda x: (x[0][0], x[0][1], x[1])))
        countries_points = self.spark.createDataFrame(rdd_points, ["country", "group", "points"])
        for group in sorted(groups):
            print("Group: {0}".format(group))
            countries_points.filter(col("group") == group).sort("points", ascending=False).select("country", "points").show()


if __name__ == "__main__":
    spark = get_spark_session("First Stage")
    years = ["2014", "2010", "2006"]
    classification_models = ["logistic_regression", "decision_tree", "random_forest"]
    accuracy_by_stage, accuracy_global = {}, {}
    for year in years:
        accuracy_by_stage[year], accuracy_global[year] = {}, {}
        print("Year: {0}".format(year))
        for classifier in classification_models:
            accuracy_by_stage[year][classifier] = {}
            print("  Classifier: {0}".format(classifier))
            for stage in sorted(get_competition_dates(year).keys()):
                accuracy_by_stage[year][classifier][stage] = (ResultStatistic(spark, year, classifier, True,
                                                                              "./test/prediction", stage=stage)
                                                              .compute_accuracy_by_stage())
            accuracy_global[year][classifier] = (ResultStatistic(spark, year, classifier, True, "./test/prediction")
                                                 .compute_accuracy_global())

    for classifier in classification_models:
        print("Classifier: {0}".format(classifier))
        for year in years:
            print("  Year: {0}".format(year))
            keys = accuracy_by_stage[year][classifier].keys()
            keys.sort()
            for key, value in sorted(list(accuracy_by_stage[year][classifier].iteritems()), key=lambda p: p[0]):
                print("   {0}: {1}".format(key, value))
    print("\n"*3)
    for year in years:
        print("  Year: {0}".format(year))
        for key, value in sorted(accuracy_global[year].iteritems(), key=lambda p: p[0]):
            print("   {0}: {1}, ".format(key, value)),
        print("")
        
    ResultStatistic(spark, "2014", "decision_tree", True, "./test/prediction").print_scores_by_group()
