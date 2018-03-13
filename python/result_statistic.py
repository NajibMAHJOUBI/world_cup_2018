# -*- coding: utf-8 -*-
import os

from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType

from get_spark_session import get_spark_session
from get_competition_dates import get_competition_dates
from classification_model import ClassificationModel

def team_result(results):
    resu = 0.0
    for tp in results:
        resu += tp[0] * tp[1]
    return resu

class ResultStatistic:

    def __init__(self, spark_session, year, stage, classifier_model, stacking, path_data):
        self.spark = spark_session
        self.year = year
        self.stage = stage
        self.classifier_model = classifier_model
        self.stacking = stacking
        self.path_data = path_data

        self.vs_groups = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]

        self.label_prediction = None

    def __str__(self):
        s = "Result Statistic\n"
        s += "Spark session: {0}\n".format(self.spark)
        s += "Year: {0}\n".format(self.year)
        s += "Satge: {0}\n".format(self.stage)
        return s

    def run(self):
        self.load_label_prediction()

    def get_path_label_prediction(self):
        if self.stacking:
            return os.path.join(self.path_data, self.year, self.stage, "stacking", self.classifier_model)
        else:
            return os.path.join(self.path_data, self.year, self.stage, self.classifier_model)

    def load_label_prediction(self):
        schema = StructType([StructField("matches", StringType(), True),
                             StructField("label", DoubleType(), True),
                             StructField("prediction", DoubleType(), True)])
        self.label_prediction = self.spark.read.csv(self.get_path_label_prediction(), header=True, sep=",",
                                                    schema=schema)

    def compute_accuracy(self):
        classification_model = ClassificationModel(None, None, None, None, None, None)
        classification_model.set_transform(self.label_prediction)
        classification_model.define_evaluator()
        return classification_model.evaluate_evaluator()

    # def load_results_top_groups(self):
    #     schema = StructType([StructField("group", StringType(), True),
    #                          StructField("1st", StringType(), True),
    #                          StructField("2nd", StringType(), True)])
    #     return self.spark.read.csv("../data/first_round_top_groups.csv", sep=",", schema=schema, header=False)
    #
    # def load_data_groups(self):
    #     udf_strip = udf(lambda x: x.strip(), StringType())
    #     schema = StructType([StructField("group", StringType(), True),
    #                          StructField("country_1", StringType(), True),
    #                          StructField("country_2", StringType(), True),
    #                          StructField("country_3", StringType(), True),
    #                          StructField("country_4", StringType(), True)])
    #     return (self.spark.read.csv("../data/groups.csv", sep=",", schema=schema, header=False)
    #             .select(col("group"),
    #                     udf_strip(col("country_1")).alias("country_1"),
    #                     udf_strip(col("country_2")).alias("country_2"),
    #                     udf_strip(col("country_3")).alias("country_3"),
    #                     udf_strip(col("country_4")).alias("country_4")))
    #
    # def win_losse_drawn_count_by_group(self, data):
    #     data.groupBy("prediction").count().show()
    #
    # def win_losse_drawn_by_team(self):
    #     rdd_team_1 = (self.data
    #                   .groupBy(["country_1", "result_team_1"]).count()
    #                   .rdd
    #                   .map(lambda x: ((x["country_1"], x["result_team_1"]), x["count"])))
    #
    #     rdd_team_2 = (self.data
    #                   .groupBy(["country_2", "result_team_2"]).count()
    #                   .rdd
    #                   .map(lambda x: ((x["country_2"], x["result_team_2"]), x["count"])))
    #
    #     teams_results =  (rdd_team_1
    #                       .union(rdd_team_2)
    #                       .reduceByKey(lambda x,y: x + y)
    #                       .map(lambda x: (x[0][0], [(x[0][1], x[1])]))
    #                       .reduceByKey(lambda x,y: x + y)
    #                       .map(lambda x: (x[0], sorted(x[1], key=lambda tup: tup[0], reverse=True))).collect())
    #
    #     return {key: value for (key,value) in teams_results}
    #
    # def global_result_by_team(self, bool_print):
    #     dic_data_groups = self.get_dic_data_groups()
    #     groups = sorted(dic_data_groups.keys())
    #     result_by_team = self.win_losse_drawn_by_team()
    #     dic_result_group_team = {group: {} for group in groups}
    #     if bool_print:
    #         for group in groups:
    #             print("Group: {0}".format(group))
    #             for country in dic_data_groups[group]:
    #                 result = team_result(result_by_team[country])
    #                 dic_result_group_team[group][country] = result
    #                 print("Country {0}: {1}".format(country, result))
    #             print("")
    #     return dic_result_group_team
    #
    # def get_dic_data_groups(self):
    #     udf_get_countries = udf(lambda country_1, country_2, country_3, country_4:
    #                             [country_1, country_2, country_3, country_4], ArrayType(StringType()))
    #     data_collect = (self.load_data_groups()
    #                     .withColumn("countries",
    #                                 udf_get_countries(col("country_1"), col("country_2"),
    #                                                   col("country_3"), col("country_4")))
    #                     .select("group", "countries")
    #                     .rdd.map(lambda x: (x["group"], x["countries"]))
    #                     .collect())
    #     return {key: value for (key, value) in data_collect}
    #
    # def first_second_by_group(self, dic_result_group_team):
    #     dic_first_by_group, dic_second_by_group = {}, {}
    #     groups = sorted(dic_result_group_team.keys())
    #     for group in groups:
    #         country_result = list(dic_result_group_team[group].iteritems())
    #         country_result.sort(key=lambda tp: tp[1], reverse=True)
    #         results = list(np.unique(map(lambda tp: tp[1], country_result)))
    #         results.sort(reverse=True)
    #
    #         first_teams = filter(lambda tp: tp[1] == results[0], country_result)
    #         dic_first_by_group[group] = map(lambda tp: tp[0], first_teams)
    #
    #         if len(results) >= 2:
    #             second_teams = filter(lambda tp: tp[1] == results[1], country_result)
    #             dic_second_by_group[group] = map(lambda tp: tp[0], second_teams)
    #         else:
    #             dic_second_by_group[group] = None
    #     return dic_first_by_group, dic_second_by_group
    #
    # def accuracy_teams_qualified(self, dic_first_by_group, dic_second_by_group):
    #     data = self.load_results_top_groups()
    #     label_1st = data.select("1st").rdd.map(lambda x: x["1st"]).collect()
    #     label_2nd = data.select("2nd").rdd.map(lambda x: x["2nd"]).collect()
    #     set_label = set(label_1st + label_2nd)
    #
    #     prediction_1st = [dic_first_by_group[key] for key in dic_first_by_group.keys()]
    #     prediction_2nd = [dic_second_by_group[key] for key in dic_second_by_group.keys()]
    #     prediction_1st = filter(lambda x: len(x) == 1, prediction_1st)
    #     prediction_2nd = filter(lambda x: len(x) == 1, prediction_2nd)
    #     prediction_1st = list(chain(*prediction_1st))
    #     prediction_2nd = list(chain(*prediction_2nd))
    #     set_prediction = set(prediction_1st + prediction_2nd)
    #
    #     accuracy_global = float(len(set_label.intersection(set_prediction))) / len(set_label)
    #     accuracy_1st = float(len(set(label_1st).intersection(set(prediction_1st)))) / len(label_1st)
    #     accuracy_2nd = float(len(set(label_2nd).intersection(set(prediction_2nd)))) / len(label_2nd)
    #
    #     return accuracy_global, accuracy_1st, accuracy_2nd


# Statistique

# Nombre d'équipes qualifiées
# Equipe qualifiées avec le bon rank (1er et 2nd)
# Nombre de matches avec les bonnes équipes détectées



if __name__ == "__main__":
    spark = get_spark_session("First Stage")
    years = ["2014", "2010", "2006"]
    accuracy = {}
    for year in years:
        accuracy[year] = {}
        print("Year: {0}".format(year))
        for stage in sorted(get_competition_dates(year).keys()):
            result_statistic = ResultStatistic(spark, year, stage, "decision_tree", True, "./test/prediction")
            result_statistic.run()
            accuracy[year][stage] = result_statistic.compute_accuracy()


for year in years:
    print("Year: {0}".format(year))
    for stage in sorted(get_competition_dates(year).keys()):
        print("  {0}: {1}".format(stage, accuracy[year][stage]))
