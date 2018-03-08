# -*- coding: utf-8 -*-
"""
Define the best model via cross validation based on the qualification phases
"""
# pyspark libraries
from pyspark.sql import SparkSession
# my libraries
from classification_model import ClassificationModel
from featurization_data import FeaturizationData

# define spark session
spark = SparkSession.builder.master("local").appName("World_Cup_2014").getOrCreate()

# featurization confederation data
confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "playoffs", "UEFA", "WCP"]
featurization = FeaturizationData(spark, confederations)
featurization.run()
data = featurization.get_data_union().cache()
#print("data count: {0}".format(data.count()))
#data.show(5)

# Classification Model
dic_evaluate_model, dic_model_classifier = {}, {}
classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron", "one_vs_rest"]
#classification_model = ["decision_tree"]
for model in classification_model:
    print("Model classification: {0}".format(model))
    classification_model = ClassificationModel(data, model, "./test/classification_model", "./test/transform", validator="train_validation", list_layers=None)
    classification_model.run()
##    dic_model_classifier[model] = classification_model.get_best_model()
    dic_evaluate_model[model] = classification_model.evaluate_evaluator()

print("Accuracy")
for key, value in dic_evaluate_model.iteritems():
    print("{0}: {1}".format(key, value))

#print(dic_model_classifier)

