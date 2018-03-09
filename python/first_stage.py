from pyspark.ml.feature import IndexToString, StringIndexerModel
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType

from classification_model import ClassificationModel
from stacking_ensemble_method import Stacking


class FirstRound:
    def __init__(self, spark_session, classification_model, stacking, data, path_model, path_prediction):
        self.spark = spark_session
        self.model_classifier = classification_model
        self.stacking_test = stacking
        self.data = data  
        self.path_model = path_model
        self.path_prediction = path_prediction

        self.label_prediction = None
        self.transform = None
    
    def __str__(self):
        pass
    
    def run(self):
        if not self.stacking_test:
            self.transform_model(self.data)
            self.save_prediction(self.transform.select("id", "label", "prediction"))
        else:
            classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron", "one_vs_rest"]
            stacking = Stacking(self.spark, classification_model, self.model_classifier, "train_validation", "./test/prediction")
            stacking.run()
            self.transform = stacking.stacking_transform()
            self.save_prediction(self.transform.select("id", "label", "prediction"))
            self.apply_intdex_to_string()

    def load_data_teams(self):
        schema = StructType([StructField("team", StringType(), True),
                             StructField("country", StringType(), True)])
        return self.spark.read.csv("./data/common/en.teams.tsv", sep="\t", header=False, schema=schema)

    def get_data(self):
        return self.data

    def get_transform(self):
        return self.transform

    def apply_index_to_string(self):
        teams = self.load_data_teams()
        labels = StringIndexerModel.load("./test/string_indexer").labels
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

    def load_classifier_model(self):
        return ClassificationModel(None, self.model_classifier, self.path_model, None).get_best_model()
        
    def transform_model(self, data):
        self.transform = self.load_classifier_model().transform(data)

    def evaluate_evaluator(self, data):
        classification_model = ClassificationModel(None, None, None, None)
        classification_model.define_evaluator()
        classification_model.set_transform(data)         
        return classification_model.evaluate_evaluator()

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
  
