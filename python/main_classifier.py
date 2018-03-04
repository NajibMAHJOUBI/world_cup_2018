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
print("data count: {0}".format(data.count()))
#data.show()
#
# Classification Model
classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron", "one_vs_rest"]
dic_evaluate_model = {}
dic_model_classifier = {}
for model in classification_model:
    print("Model classification: {0}".format(model))
    classification_model = ClassificationModel(data, model, "train_validation")
    classification_model.run()
    dic_model_classifier[model] = classification_model.get_best_model()
    dic_evaluate_model[model] = classification_model.evaluate_evaluator()
print(dic_evaluate_model)
print(dic_model_classifier)


# World Cup 2014
#featurization = FeaturizationData(spark, None)
#data = featurization.get_qualifying_start_data("WCF")
#data.show(10)

#schema = StructType([
#	    StructField("group", StringType(), True),
#	    StructField("country_1", StringType(), True),
#	    StructField("country_2", StringType(), True)])

#first_round_matches = spark.read.csv("./data/first_round_matches", sep=",", header=False, schema=schema)
#first_round_matches.show(10)

#schema = StructType([
#    StructField("team", StringType(), True),
#    StructField("country", StringType(), True)])
#teams = spark.read.csv("./data/common/en.teams.tsv", sep="\t", header=False, schema=schema)
#teams.show(10)

#udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())

#features = first_round_matches\
#.join(teams, col("country_1") == col("country"))\
#.drop("country").drop("country_1").withColumnRenamed("team", "team_1")\
#.join(teams, col("country_2") == col("country"))\
#.drop("country").drop("country_2").withColumnRenamed("team", "team_2")\
#.join(data, col("team_1") == col("team"))\
#.drop("team").withColumnRenamed("features", "features_1")\
#.join(data, col("team_2") == col("team"))\
#.drop("team").withColumnRenamed("features", "features_2")\
#.withColumn("features", udf_diff_features(col("features_1"), col("features_2")))

#udf_tuple_matches = udf(lambda team_1,team_2: (team_1, team_2), ArrayType(StringType()))
#label = featurization.get_qualifying_results_data("WCF").withColumn("matches", udf_tuple_matches(col("team_1"), col("team_2")))
#label.show(5)

#world_cup_accuracy = []
#for model_classifier in dic_model_classifier.keys():
#    model = dic_model_classifier[model_classifier]
#    prediction = model.transform(features)\
#                      .withColumn("matches", udf_tuple_matches(col("team_1"), col("team_2")))\
#                      .select("group", "matches", "prediction")
#    #prediction.show(5)
#    label_prediction = label.join(prediction, on="matches").select("label", "prediction")
#    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
#    world_cup_accuracy.append((model_classifier, evaluator.evaluate(label_prediction)))
#    print("\n\nLogistic Regression Accuracy: {0} \n\n".format(evaluator.evaluate(label_prediction)))
#
#print("\n\n\n First Round, accuracy for win/losse/draw matches:")
#print(world_cup_accuracy)

