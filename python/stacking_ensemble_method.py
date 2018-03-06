
import os
import itertools

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructField, StructType, StringType, FloatType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT

from classification_model import ClassificationModel

class Stacking:
    def __init__(self, spark_session, available_classifier, stacking_classifier, validator):
        self.spark = spark_session
        self.classifier_available = available_classifier
        self.classifier_stacking = stacking_classifier # "decision_tree", "logistic_regression", "decision_tree", "random_forest", "one_vs_rest"
        self.validator = validator
        self.path = path = "./test/prediction"

    def __str__(self):
        s = "Stacking model\n"
        s += "Available classifier: {0}".format(self.classifier_available)
        s += "Stacking classifier: {0}".format(self.stacking_classifier)
        return s

    def run(self):
        self.merge_all_data()
        new_data = self.create_features(self.data)
        self.stacking_models(new_data)
        self.stacking_transform(new_data).show(10)

    def get_schema(self):
        return (StructType([
                         StructField("team_1", StringType(), False),
                         StructField("team_2", StringType(), False),
                         StructField("prediction", FloatType(), False),
                         StructField("label", FloatType(), False)]))

    def get_indexer(self):
        return StringIndexer(inputCol="matches", outputCol="id")

    def load_all_data(self):
        udf_teams = udf(lambda team_1, team_2: "/".join([team_1, team_2]), StringType())
        dic_data = {}
        data = (self.spark.read.csv(os.path.join(self.path, self.classifier_available[0]), sep=",", header=True, schema=self.get_schema())
                .withColumn("matches", udf_teams(col("team_1"), col("team_2")))
                .withColumnRenamed("prediction", self.classifier_available[0]))
     
        matches_indexer = self.get_indexer().fit(data)

        dic_data[self.classifier_available[0]] = matches_indexer.transform(data)

        for classifier in self.classifier_available[1:]:
            data = (self.spark.read.csv(os.path.join(self.path, classifier), sep=",", header=True, schema=self.get_schema())
                    .withColumn("matches", udf_teams(col("team_1"), col("team_2")))
                    .drop("team_1").drop("team_2").drop("label")
                    .withColumnRenamed("prediction", classifier))     
            dic_data[classifier] = matches_indexer.transform(data).drop("matches")

        return dic_data

    def merge_all_data(self):
        dic_data = self.load_all_data()
        keys = dic_data.keys()
        self.data = dic_data[keys[0]]
        for key in keys[1:]:
            self.data = self.data.join(dic_data[key], on="id")
        self.data.count()
        return self.data

    def create_features(self, data):
        if (len(self.classifier_available) == 2):
            udf_features = udf(lambda x,y: Vectors.dense([x, y]), VectorUDT())
            data = data.withColumn("features", udf_features(col(self.classifier_models[0]), col(self.classifier_models[1])))
        elif (len(self.classifier_available) == 3):
            udf_features = udf(lambda x,y,z: Vectors.dense([x, y, z]), VectorUDT())
            data = data.withColumn("features", udf_features(col(self.classifier_available[0]), 
                                                            col(self.classifier_available[1]), 
                                                            col(self.classifier_available[2])))
        elif (len(self.classifier_available) == 4):
            udf_features = udf(lambda t,x,y,z: Vectors.dense([t, x, y, z]), VectorUDT())
            data = data.withColumn("features", udf_features(col(self.classifier_available[0]), 
                                                            col(self.classifier_available[1]), 
                                                            col(self.classifier_available[2]),
                                                            col(self.classifier_available[3])))
        elif (len(self.classifier_available) == 5):
            udf_features = udf(lambda s,t,x,y,z: Vectors.dense([s, t, x, y, z]), VectorUDT())
            data = data.withColumn("features", udf_features(col(self.classifier_available[0]), 
                                                            col(self.classifier_available[1]), 
                                                            col(self.classifier_available[2]),
                                                            col(self.classifier_available[3]),
                                                            col(self.classifier_available[4])))

        for classifier in self.classifier_available: data = data.drop(classifier)
        return data.drop("matches").drop("team_1").drop("team_2")


    def stacking_models(self, data):
        classification_model = ClassificationModel(data, self.classifier_stacking, 
                                                   "./test/stacking", 
                                                   validator="train_validation", 
                                                   list_layers=[[5, 3], [5, 4, 3], [5, 7, 7,5,7,5,3]])
        classification_model.run()
        self.stacking_model = classification_model.get_best_model()

    def stacking_transform(self, data):
        return self.stacking_model.transform(data) 



#    def loop_combination_classifier(self):
#        dic_accuracy = {}
#        for combination in self.combinations_classifier_features():
#            print("Combination: {0}".format(combination))  
#            new_data = self.create_features(self.data, combination)
#            accuracy = self.stacking_models(new_data)
#            dic_accuracy[combination] = accuracy
#        print(dic_accuracy)

