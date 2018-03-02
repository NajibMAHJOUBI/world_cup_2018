# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# pyspark libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, FloatType, ArrayType
from pyspark.sql.functions import col, udf, when
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
# python libraries
import numpy as np

# define spark session
spark = SparkSession.builder.master("local").appName("World_Cup_2014").getOrCreate()

# user defined function
def convert_string_to_float(x):
    x_replace_minus = x.replace(u'\u2212', '-')
    if x_replace_minus == '-':
        return np.nan
    else:
        return float(x_replace_minus)

udf_convert_string_to_float = udf(lambda x: convert_string_to_float(x), FloatType())

udf_get_percentage_game = udf(lambda x, y: x / y, FloatType())

udf_create_features = udf(lambda s,t,u,v,w,x,y,z: Vectors.dense([s,t,u,v,w,x,y,z]), VectorUDT())

udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())

def get_date_string(date, month, year):
    return year + "/" + month + "/" + date

udf_get_date_string = udf(lambda date, month, year: get_date_string(date, month, year), StringType())

def win_team_1(score_team_1, score_team_2):
    if score_team_1 > score_team_2:
        return 2.0
    elif score_team_1 < score_team_2:
        return 1.0
    else:
        return 0.0
    
udf_win_team_1 = udf(lambda team_1, team_2: win_team_1(team_1, team_2), FloatType())

# Class Classification Model
class ClassificationModel:
    def __init__(self, data, classification_model, validator=None):
        self.data = data
        self.model = classification_model
        self.validator= validator
        self.featuresCol = "features"
        self.labelCol = "label"
        self.predictionCol = "prediction"
 
    def __str__(self):
        s = "Classification model: {0}".format(self.model)
        return s

    def run(self):
       self.get_estimator()
       self.param_grid_builder()
       
       self.get_evaluator()
       self.get_validator()
       self.fit_validator()
       self.evaluate_evaluator()

    def param_grid_builder(self):
        if (self.model =="logistic_regression"):
            self.grid = ParamGridBuilder()\
                       .addGrid(self.estimator.maxIter, [10, 15, 20])\
                       .addGrid(self.estimator.regParam, [0.0, 0.1, 0.5, 1.0])\
                       .addGrid(self.estimator.elasticNetParam, [0.0, 0.1, 0.5, 1.0])\
                       .build()    
        elif(self.model == "decision_tree"):
            self.grid = ParamGridBuilder()\
                        .addGrid(self.estimator.maxDepth, [5, 10, 20])\
                        .addGrid(self.estimator.maxBins, [8, 16, 32])\
                        .build()
        elif(self.model == "random_forest"):
            self.grid = ParamGridBuilder()\
                        .addGrid(self.estimator.numTrees, [3, 6, 18])\
                        .addGrid(self.estimator.maxDepth, [5, 10, 15])\
                        .build()
        elif(self.model == "multilayer_perceptron"):
            pass
        elif(self.model == "one_vs_rest"):
            pass

    def get_estimator(self):
        if (self.model == "logistic_regression"):
            self.estimator = LogisticRegression(featuresCol=self.featuresCol, labelCol=self.labelCol, family="multinomial")
        elif(self.model == "decision_tree"):
            self.estimator = DecisionTreeClassifier(featuresCol=self.featuresCol, labelCol=self.labelCol)
        elif(self.model == "random_forest"):
            self.estimator = RandomForestClassifier(featuresCol=self.featuresCol, labelCol=self.labelCol)
        elif(self.model == "multilayer_perceptron"):
            pass
        elif(self.model == "one_vs_rest"):
            pass
    
    def get_evaluator(self):
        self.evaluator = MulticlassClassificationEvaluator(predictionCol=self.predictionCol, labelCol=self.labelCol, metricName="accuracy")

    def get_validator(self):
        if (self.validator == "cross_validation"):
            self.validation = CrossValidator(estimator=self.estimator, estimatorParamMaps=self.grid, evaluator=self.evaluator, numFolds=4)
        elif (self.validator == "train_validation"):
            self.validation = TrainValidationSplit(estimator=self.estimator, estimatorParamMaps=self.grid, evaluator=self.evaluator, trainRatio=0.75)
        else:
            self.train, self.test = self.data.randomSplit([0.8, 0.2])

    def fit_validator(self):
        if (self.validator is None):
            self.model = self.estimator.setMaxIter(20).setRegParam(0.0).fit(self.train)
        else:
            self.model = self.validation.fit(self.data)

    def transform_model(self, data):
        return self.model.transform(data)

    def evaluate_evaluator(self):
        if (self.validator is None):
            train_prediction = self.transform_model(self.train)
            test_prediction = self.transform(self.test)
            print("Accuracy on the train dataset: {0}".format(self.evaluator.evaluate(train_prediction)))
            print("Accuracy on the test dataset: {0}".format(self.evaluator.evaluate(test_prediction)))
        else:
            prediction = self.transform_model(self.data)      
            print("Accuracy on the train dataset: {0}".format(self.evaluator.evaluate(prediction)))



confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "playoffs", "UEFA"]

# FEATURIZATION DATASET
schema_qualifying_start = StructType([
    StructField("rankGroup_local", StringType(), True),
    StructField("rankGroup_global", StringType(), True),
    StructField("teamGroup_team", StringType(), True),
    StructField("ratingGroup_rating", StringType(), True),
    StructField("highestGroup_rank_max", StringType(), True),
    StructField("highestGroup_rating_max", StringType(), True),
    StructField("averageGroup_rank_avg", StringType(), True),
    StructField("averageGroup_rating_avg", StringType(), True),
    StructField("lowestGroup_rank_min", StringType(), True),
    StructField("lowestGroup_rating_min", StringType(), True),
    StructField("change3mGroup_rank_three_month_change", StringType(), True),
    StructField("change3mGroup_rating_three_month_change", StringType(), True),
    StructField("change6mGroup_rank_six_month_change", StringType(), True),
    StructField("change6mGroup_rating_six_month_change", StringType(), True),
    StructField("change1yGroup_rank_one_year_change", StringType(), True),
    StructField("change1yGroup_rating_one_year_change", StringType(), True),
    StructField("change2yGroup_rank_two_year_change", StringType(), True),
    StructField("change2yGroup_rating_two_year_change", StringType(), True),
    StructField("change5yGroup_rank_five_year_change", StringType(), True),
    StructField("change5yGroup_rating_five_year_change", StringType(), True),
    StructField("change10yGroup_rank_ten_year_change", StringType(), True),
    StructField("change10yGroup_rating_ten_year_change", StringType(), True),
    StructField("matchesGroup_total", StringType(), True),
    StructField("matchesGroup_home", StringType(), True),
    StructField("matchesGroup_away", StringType(), True),
    StructField("matchesGroup_neutral", StringType(), True),
    StructField("matchesGroup_wins", StringType(), True),
    StructField("matchesGroup_losses", StringType(), True),
    StructField("matchesGroup_draws", StringType(), True),
    StructField("goalsGroup_for", StringType(), True),
    StructField("goalsGroup_against", StringType(), True)
])

names_start_to_convert = schema_qualifying_start.names
names_start_to_convert.remove("teamGroup_team")

schema_qualifying_results = StructType([
    StructField("year", StringType(), True),
    StructField("month", StringType(), True),
    StructField("date", StringType(), True),
    StructField("team_1", StringType(), True),
    StructField("team_2", StringType(), True),
    StructField("score_team_1", IntegerType(), True),
    StructField("score_team_2", IntegerType(), True),
    StructField("tournament", StringType(), True),
    StructField("country_played", StringType(), True),
    StructField("rating_moved", StringType(), True),
    StructField("rating_team_1", StringType(), True),
    StructField("rating_team_2", StringType(), True),
    StructField("rank_moved_team_1", StringType(), True),
    StructField("rank_moved_team_2", StringType(), True),
    StructField("rank_team_1", StringType(), True),
    StructField("rank_team_2", StringType(), True)
])

names_results_to_convert = schema_qualifying_results.names
names_results_to_remove = ["date",  "team_1", "team_2", "score_team_1", "score_team_2", "tournament", "country_played"]
for name in names_results_to_remove: names_results_to_convert.remove(name)

dic_data = {}
for confederation in confederations:
    path = "./data/{0}/2014_World_Cup_{1}_qualifying_start.tsv".format(confederation, confederation)
    df_qualifying_start = spark.read.csv(path, sep="\t", 
                                      schema=schema_qualifying_start, header=False)\
                                 .select([udf_convert_string_to_float(col(name)).alias(name) for name in names_start_to_convert] + ["teamGroup_team"])\
                                 .withColumn("features", udf_create_features(
                                             udf_get_percentage_game(col("matchesGroup_home"), col("matchesGroup_total")), 
                                             udf_get_percentage_game(col("matchesGroup_away"), col("matchesGroup_total")),
                                             udf_get_percentage_game(col("matchesGroup_neutral"), col("matchesGroup_total")),
                                             udf_get_percentage_game(col("matchesGroup_wins"), col("matchesGroup_total")), 
                                             udf_get_percentage_game(col("matchesGroup_losses"), col("matchesGroup_total")), 
                                             udf_get_percentage_game(col("matchesGroup_draws"), col("matchesGroup_total")),
                                             udf_get_percentage_game(col("goalsGroup_for"), col("matchesGroup_total")),  
                                             udf_get_percentage_game(col("goalsGroup_against"), col("matchesGroup_total"))))\
                                  .withColumnRenamed("teamGroup_team", "team")\
                                  .select("team", "features")

    path = "./data/{0}/2014_World_Cup_{1}_qualifying_results.tsv".format(confederation, confederation)
    df_qualifying_results = spark.read.csv(path, sep="\t", schema=schema_qualifying_results, header=False)\
                              .select([udf_convert_string_to_float(col(name)).alias(name) for name in names_results_to_convert] + names_results_to_remove)\
                              .select("team_1", "team_2", "score_team_1", "score_team_2")\
                              .withColumn("label", udf_win_team_1(col("score_team_1"), col("score_team_2")))\
                              .select("team_1", "team_2", "label")\

   
    data = df_qualifying_results.join(df_qualifying_start, df_qualifying_results.team_1 == df_qualifying_start.team)\
                                 .withColumnRenamed("features", "features_1").drop("team")\
                                 .join(df_qualifying_start, df_qualifying_results.team_2 == df_qualifying_start.team)\
                                 .withColumnRenamed("features", "features_2").drop("team")\
                                 .withColumn("features", udf_diff_features(col("features_1"), col("features_2")))\
                                 .select("label", "features")

    dic_data[confederation] = data


for confederation in confederations:
    print(confederation, dic_data[confederation].count())


print("")
classification_model = ["logistic_regression", "decision_tree", "random_forest"]
for model in classification_model:
    print("Model classification: {0}".format(model))
    ClassificationModel(dic_data["AFC"], model, "train_validation").run()




