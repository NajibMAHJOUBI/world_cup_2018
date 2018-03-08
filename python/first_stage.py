
import numpy as np

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, BooleanType, FloatType
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf, col
from pyspark.ml.feature import StringIndexer, IndexToString, StringIndexerModel

from itertools import combinations
from featurization_data import FeaturizationData
from classification_model import ClassificationModel
from stacking_ensemble_method import Stacking
from print_result import first_second_by_group, print_first_second_by_group, print_matches_next_stage

def team_result(results):
    resu = 0.0
    for tp in results:
        resu += tp[0] * tp[1]
    return resu

class FirstRound:
    def __init__(self, spark_session, classification_model, stacking, data, path_model, path_prediction):
        self.spark = spark_session
        self.model_classifier = classification_model
        self.test_stacking = stacking
        self.data = data  
        self.path_model = path_model
        self.path_prediction = path_prediction
        self.tp_groups = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    
    def __str__(self):
        pass
    
    def run(self):
        if (not self.test_stacking):
#            self.load_data_teams()
#            label = self.get_matches_results(self.start_date, self.end_date)
#            label.show(5)
#            matches_features = self.define_matches_features(label)
#            matches_features.show(5)
            self.transform_model(self.data)
#            self.transform.show()
            self.save_prediction(self.transform.select("id", "label", "prediction"))
#            self.join_label_prediction(label, prediction)
#            self.label_prediction.show(5)
#            self.save_prediction(self.label_prediction)
        else:
            classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron", "one_vs_rest"]
            stacking = Stacking(self.spark, classification_model, self.model_classifier, "train_validation", "./test/prediction")
            stacking.run()
            self.transform = stacking.stacking_transform()
            self.save_prediction(self.transform.select("id", "label", "prediction"))
            self.apply_intdex_to_string()
            
#            self.win_losse_drawn_count_by_group()
            dic_result_group_team = self.global_result_by_team()
            dic_first_by_group, dic_second_by_group = self.first_second_by_group(dic_result_group_team)
            print(dic_first_by_group)
            print(dic_second_by_group)
            print_first_second_by_group(dic_first_by_group, dic_second_by_group)
            print_matches_next_stage(dic_first_by_group, dic_second_by_group)

    def load_data_teams(self):
        schema = StructType([
            StructField("team", StringType(), True),
            StructField("country", StringType(), True)])
        return self.spark.read.csv("./data/common/en.teams.tsv", sep="\t", header=False, schema=schema)

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

    def get_string_indexer(self):
        return StringIndexerModel.load("./test/string_indexer")

    def apply_intdex_to_string(self):
        teams = self.load_data_teams()
        labels = self.get_string_indexer().labels
        model = IndexToString(inputCol="id", outputCol="matches", labels=labels)



        udf_get_team_1 = udf(lambda x: x.split("_")[0].split("/")[0], StringType())
        udf_get_team_2 = udf(lambda x: x.split("_")[0].split("/")[1], StringType())
        udf_get_date = udf(lambda x: x.split("_")[1], StringType())         


        def result_team_2(result):
            if (result == 2):
                return 1.0
            elif (result == 1):
                return 2.0
            else:
                return 0.0

        udf_result_team_2 = udf(lambda result: result_team_2(result), FloatType())

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

    def get_transform(self):
        return self.transform

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

    def load_classifier_model(self):
        return ClassificationModel(None, self.model_classifier, self.path_model, None).get_best_model()
        
    def transform_model(self, data):
        self.transform = self.load_classifier_model().transform(data)

    def evaluate_evaluator(self, data):
        classification_model = ClassificationModel(None, None, None, None)
        classification_model.define_evaluator()
        classification_model.set_transform(data)         
        return classification_model.evaluate_evaluator()

    def win_losse_drawn_count_by_group(self, data):
        data.groupBy("prediction").count().show()

    def win_losse_drawn_by_team(self):
        rdd_team_1 = (self.data
                      .groupBy(["country_1", "result_team_1"]).count()
                      .rdd
                      .map(lambda x: ((x["country_1"], x["result_team_1"]), x["count"])))

        rdd_team_2 = (self.data
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

    def save_matches_next_stage(self, dic_first_by_group, dic_second_by_group):
        data = []
        for tp in self.tp_groups:
            data.append(('/'.join(dic_first_by_group[tp[0]]), '/'.join(dic_second_by_group[tp[1]]))) 
            data.append(('/'.join(dic_first_by_group[tp[1]]), '/'.join(dic_second_by_group[tp[0]])))
        (self.spark.createDataFrame(data, ["team_1", "team_2"])
         .write
         .csv("./test/matches_next_stage/{0}".format(self.model_classifier), mode="overwrite", sep=",", header=True))

    def save_prediction(self, prediction):
        (prediction
         .coalesce(1)
         .write.csv("{0}/{1}".format(self.path_prediction, self.model_classifier), mode="overwrite", sep=",", header=True))

    def get_dic_data_groups(self):
        udf_get_countries = udf(lambda country_1, country_2, country_3, country_4: [country_1, country_2, country_3, country_4], ArrayType(StringType()))
        data_collect = (self.load_data_groups()
                        .withColumn("countries", udf_get_countries(col("country_1"),col("country_2"),col("country_3"),col("country_4")))
                        .select("group", "countries")
                        .rdd.map(lambda x: (x["group"], x["countries"]))
                        .collect())
        return {key: value for (key,value) in data_collect}

    def global_result_by_team(self):    
        dic_data_groups = self.get_dic_data_groups()
        groups = sorted(dic_data_groups.keys())
        result_by_team = self.win_losse_drawn_by_team()
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
        return dic_first_by_group, dic_second_by_group      
