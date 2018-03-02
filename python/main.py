# -*- coding: utf-8 -*-
"""
Define the best model via cross validation based on the qualification phases
"""
# pyspark libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, FloatType, ArrayType
from pyspark.sql.functions import col, udf, when
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row
# python libraries
import numpy as np
# my libraries
from classification_model import ClassificationModel
from featurization_data import FeaturizationData

# define spark session
spark = SparkSession.builder.master("local").appName("World_Cup_2014").getOrCreate()
confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "playoffs", "UEFA", "WCP"]

# featurization confederation data
featurization = FeaturizationData(spark, confederations)
featurization.run()
data = featurization.get_data_union().cache()
print("data count: {0}".format(data.count()))
#data.show()

# Classification Model
print("")
classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron"]
evaluate_model = []
for model in classification_model:
    print("Model classification: {0}".format(model))
    classification_model = ClassificationModel(data, model, "train_validation")
    classification_model.run()
    evaluate_model.append((model, classification_model.evaluate_evaluator()))


print(evaluate_model)

