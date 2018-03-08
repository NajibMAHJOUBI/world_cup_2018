# -*- coding: utf-8 -*-
"""
First Round World Cup
"""
# pyspark libraries
from pyspark.sql import SparkSession
# my libraries
from featurization_data import FeaturizationData
from first_stage import FirstRound
from result_statistic import ResultStatistic

# define spark session
spark = SparkSession.builder.master("local").appName("World_Cup_2014").getOrCreate()

# Play first round of the world cup
#
# featurization confederation data
featurization = FeaturizationData(spark, ["WCF"], list_date=["2014/06/12", "2014/06/26"])
featurization.run()
data = featurization.get_data_union().cache()
#data.show(10)
print(data.count())

# --> Elementary classifier method
#classification_model = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron", "one_vs_rest"]
##classification_model = ["decision_tree"]
#first_stage_accuracy = {}
#for model in classification_model:
#    print("\n\nModel classifier: {0}".format(model))
#    first_round = FirstRound(spark, model, False, data, "./test/classification_model", "./test/prediction")
#    first_round.run()
#    first_stage_accuracy[model] = first_round.evaluate_evaluator(first_round.get_transform())
#    
#for key, value in first_stage_accuracy.iteritems():
#    print("{0}: {1}".format(key, value))

# --> Stacking method
first_round = FirstRound(spark, "random_forest", True, data, "./test/classification_model/stacking", "./test/prediction/stacking")
first_round.run()
print("Stacking: {0}".format(first_round.evaluate_evaluator(first_round.get_transform())))

#first_round.get_data().show()


result_stat = ResultStatistic(spark, first_round.get_data())
print(result_stat)
result_stat.run()
