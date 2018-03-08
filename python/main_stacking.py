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
stacking = Stacking(spark, classification_model, "decision_tree", "train_validation", "./test/transform")
stacking.run()

#def combinations_classifier_features(self):
#    number_classifier = len(self.classifier_models)
#    combinations = []
#    for nb in range(2, number_classifier+1):
#        combinations += list(itertools.combinations(self.classifier_models, nb))
#    return combinations
