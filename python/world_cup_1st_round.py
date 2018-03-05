
import numpy as np

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, BooleanType, FloatType
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf, col

from itertools import combinations
from featurization_data import FeaturizationData
from classification_model import ClassificationModel

def team_result(results):
    resu = 0.0
    for tp in results:
        resu += tp[0] * tp[1]
    return resu

class WorldCupFirstRound:
    def __init__(self, spark_session, classification_model, round_start_date, round_end_date):
        self.spark = spark_session
        self.model_classifier = classification_model
        self.start_date = round_start_date
        self.end_date = round_end_date
    
    def __str__(self):
        pass
    
    def run(self):
        self.load_data_teams()
        label = self.get_matches_results(self.start_date, self.end_date)
        matches_features = self.define_matches_features(label)
        prediction = self.transform_model(matches_features)
        label_prediction = self.join_label_prediction(label, prediction)
        print("Accuracy: {0}".format(self.evaluate_evaluator(label_prediction)))
        self.win_losse_drawn_count_by_group(label_prediction)
#        dic_result_group_team = self.global_result_by_team(label_prediction)
#        self.first_second_by_group(dic_result_group_team)
    
    def load_data_groups(self):
        udf_strip = udf(lambda x: x.strip(), StringType())
        schema = StructType([
            StructField("group", StringType(), True),
            StructField("country_1", StringType(), True),
            StructField("country_2", StringType(), True),
            StructField("country_3", StringType(), True),
            StructField("country_4", StringType(), True)])
        return (self.spark.read.csv("./data/groups.csv", sep=",", schema=schema, header=False)
                .select(col("group"), udf_strip(col("country_1")).alias("country_1"), 
                                      udf_strip(col("country_2")).alias("country_2"),
                                      udf_strip(col("country_3")).alias("country_3"),                                      
                                      udf_strip(col("country_4")).alias("country_4")))

    def load_data_teams(self):
        schema = StructType([
            StructField("team", StringType(), True),
            StructField("country", StringType(), True)])
        self.teams = self.spark.read.csv("./data/common/en.teams.tsv", sep="\t", header=False, schema=schema)

    def get_dic_data_groups(self):
        udf_get_countries = udf(lambda country_1, country_2, country_3, country_4: [country_1, country_2, country_3, country_4], ArrayType(StringType()))
        data_collect = (self.load_data_groups()
                        .withColumn("countries", udf_get_countries(col("country_1"),col("country_2"),col("country_3"),col("country_4")))
                        .select("group", "countries")
                        .rdd.map(lambda x: (x["group"], x["countries"]))
                        .collect())
        return {key: value for (key,value) in data_collect}

        
    def get_team_features(self):
        return FeaturizationData(self.spark, None).get_qualifying_start_data("WCF")

    def get_matches_results(self, start_date, end_date):
        def filter_date(date, start_date, end_date):
            if ((date >= start_date) and (date <= end_date)):
                return True
            else:
                return False
        udf_filter_date = udf(lambda date: filter_date(date, start_date, end_date), BooleanType())
        return (FeaturizationData(self.spark, None).get_qualifying_results_data("WCF")
                .filter(udf_filter_date(col("date"))))

    def join_label_prediction(self, label, prediction):
        udf_tuple_matches = udf(lambda team_1,team_2: (team_1, team_2), ArrayType(StringType()))
        label = label.withColumn("matches", udf_tuple_matches(col("team_1"), col("team_2")))
        prediction = (prediction
                      .withColumn("matches", udf_tuple_matches(col("team_1"), col("team_2")))
                      .select("matches", "prediction"))
        return label.join(prediction, on="matches").select("label", "prediction", "matches")

    def define_matches_features(self, data):
        udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())
        team_features = self.get_team_features()
        return (data
                .join(team_features, col("team_1") == col("team"))
                .drop("team").withColumnRenamed("features", "features_1")
                .join(team_features, col("team_2") == col("team"))
                .drop("team").withColumnRenamed("features", "features_2")
                .withColumn("features", udf_diff_features(col("features_1"), col("features_2"))))

    def load_classifier_model(self):
        return ClassificationModel(None, self.model_classifier).get_best_model()
        
    def transform_model(self, data):
        return self.load_classifier_model().transform(data)

    def evaluate_evaluator(self, label_prediction):
        return ClassificationModel(None, None).get_evaluator().evaluate(label_prediction)

    def win_losse_drawn_count_by_group(self, data):
        data.groupBy("prediction").count().show()

    def win_losse_drawn_by_team(self, data):
        udf_get_team_1 = udf(lambda x: x[0], StringType())
        udf_get_team_2 = udf(lambda x: x[1], StringType())

        def result_team_2(result):
            if (result == 2):
                return 1.0
            elif (result == 1):
                return 2.0
            else:
                return 0.0

        udf_result_team_2 = udf(lambda result: result_team_2(result), FloatType())

        data = (data
                .withColumn("team_1", udf_get_team_1(col("matches")))
                .withColumn("team_2", udf_get_team_2(col("matches")))
                .withColumn("result_team_2", udf_result_team_2(col("prediction")))
                .withColumnRenamed("prediction", "result_team_1"))

        data = (data.join(self.teams, data.team_1 == self.teams.team)
                .withColumnRenamed("country", "country_1").drop("team")
                .join(self.teams, data.team_2 == self.teams.team)
                .withColumnRenamed("country", "country_2").drop("team"))

        rdd_team_1 = (data
                      .groupBy(["country_1", "result_team_1"]).count()
                      .rdd
                      .map(lambda x: ((x["country_1"], x["result_team_1"]), x["count"])))

        rdd_team_2 = (data
                      .groupBy(["country_2", "result_team_2"]).count()
                      .rdd
                      .map(lambda x: ((x["country_2"], x["result_team_2"]), x["count"])))

        teams_results =  (rdd_team_1
                          .union(rdd_team_2)
                          .reduceByKey(lambda x,y: x + y)
                          .map(lambda x: (x[0][0], [(x[0][1], x[1])]))
                          .reduceByKey(lambda x,y: x + y)
                          .map(lambda x: (x[0], sorted(x[1], key=lambda tup: tup[0], reverse=True))).collect())

        return {key: value for (key,value) in teams_results}

    def global_result_by_team(self, data):    
        dic_data_groups = self.get_dic_data_groups()
        groups = sorted(dic_data_groups.keys())
        result_by_team = self.win_losse_drawn_by_team(data)
        dic_result_group_team = {group: {} for group in groups}
        for group in groups:
            print("Group: {0}".format(group))
            for country in dic_data_groups[group]:
                result = team_result(result_by_team[country])
                dic_result_group_team[group][country] = result
                print("Country {0}: {1}".format(country, result))
            print("")
        return dic_result_group_team

    def first_second_by_group(self, dic_result_group_team):
        dic_first_by_group, dic_second_by_group = {}, {}
        groups = sorted(dic_result_group_team.keys())
        for group in groups:
            country_result = list(dic_result_group_team[group].iteritems())
            country_result.sort(key=lambda tp: tp[1], reverse=True)
            results = list(np.unique(map(lambda tp: tp[1], country_result)))
            results.sort(reverse=True)
            
            first_teams = filter(lambda tp: tp[1] == results[0], country_result)
            dic_first_by_group[group] = map(lambda tp: tp[0], first_teams)

            if (len(results) >= 2):
                second_teams = filter(lambda tp: tp[1] == results[1], country_result)
                dic_second_by_group[group] = map(lambda tp: tp[0], second_teams)
            else:
                dic_second_by_group[group] = None

        for group in groups:
            print("Group: {0}".format(group))
            print("1st: {0}".format(dic_first_by_group[group]))
            print("2nd: {0}".format(dic_second_by_group[group]))




