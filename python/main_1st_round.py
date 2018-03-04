# -*- coding: utf-8 -*-
"""
First Round World Cup
"""
# pyspark libraries
from pyspark.sql import SparkSession
# my libraries
from featurization_data import FeaturizationData
from world_cup_1st_round import WorldCupFirstRound


# define spark session
spark = SparkSession.builder.master("local").appName("World_Cup_2014").getOrCreate()

# Play first round of the world cup
#classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron", "one_vs_rest"]
classification_model = ["decision_tree"]
for model in classification_model:
    print("\n\nModel classifier: {0}".format(model))
    first_round = WorldCupFirstRound(spark, model, "2014/06/12", "2014/06/26")
    first_round.run()
