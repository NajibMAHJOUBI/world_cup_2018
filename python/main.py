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
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.tuning import ParamGridBuilder
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

confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "playoffs", "UEFA"]

# qualifying start data set by confederation
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

names_to_convert = schema_qualifying_start.names
names_to_convert.remove("teamGroup_team")

dic_qualifying_start = {}
for confederation in confederations:
    path = "./data/{0}/2014_World_Cup_{1}_qualifying_start.tsv".format(confederation, confederation)
    dic_qualifying_start[confederation] = spark.read.csv(path, sep="\t", 
                                      schema=schema_qualifying_start, header=False)\
                                 .select([udf_convert_string_to_float(col(name)).alias(name) for name in names_to_convert] + ["teamGroup_team"])\
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


# qualifying result 
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

names_to_convert = schema_qualifying_results.names
names_to_remove = ["date",  "team_1", "team_2", "score_team_1", "score_team_2", "tournament", "country_played"]
for name in names_to_remove: names_to_convert.remove(name)

dic_qualifying_results = {}
for confederation in confederations:
    path = "./data/{0}/2014_World_Cup_{1}_qualifying_results.tsv".format(confederation, confederation)
    dic_qualifying_results[confederation] = spark.read.csv(path, sep="\t", schema=schema_qualifying_results, header=False)\
                              .withColumn("new_date", udf_get_date_string(col("date"), col("month"), col("year")))\
                              .drop("date").drop("month").drop("year").withColumnRenamed("new_date", "date")\
#                              .select([udf_convert_string_to_float(col(name)).alias(name) for name in names_to_convert] + names_to_remove)\
#                              .select("team_1", "team_2", "score_team_1", "score_team_2")\
#                              .withColumn("label", udf_win_team_1(col("score_team_1"), col("score_team_2")))\
#                              .select("team_1", "team_2", "label")\
#                              .join(AFC_qualifying_start, AFC_qualifying_results.team_1 == AFC_qualifying_start.team)\
#                              .withColumnRenamed("features", "features_1").drop("team")\
#                              .join(AFC_qualifying_start, AFC_qualifying_results.team_2 == AFC_qualifying_start.team)\
#                              .withColumnRenamed("features", "features_2").drop("team")\
#                              .withColumn("features", udf_diff_features(col("features_1"), col("features_2")))\
#                              .select("label", "features")



dic_qualifying_results["AFC"].printSchema()

