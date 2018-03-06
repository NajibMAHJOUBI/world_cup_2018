# -*- coding: utf-8 -*-
"""
First Round World Cup
"""
# pyspark libraries
from pyspark.sql import SparkSession

from stacking_ensemble_method import Stacking

# define spark session
spark = SparkSession.builder.master("local").appName("World_Cup_2014").getOrCreate()

# Stacking classification models
classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron", "one_vs_rest"]
stacking = Stacking(spark, classification_model, "train_validation")
stacking.run()

