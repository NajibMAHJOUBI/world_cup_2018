
import os

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructField, StructType, StringType, FloatType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT

from classification_model import ClassificationModel

class Stacking:
    def __init__(self, spark_session, list_classifier, validator):
        self.spark = spark_session
        self.classifier_models = list_classifier
        self.path = path = "./test/prediction"
        self.validator = validator

    def __str__(self):
        pass

    def run(self):
        data = self.merge_all_data()
#        print("Data count: {0}".format(data.count()))
#        data.show(5)
        data = self.create_features(data)
#        data.show(5)
        self.stacking_models(data)

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
        data = (self.spark.read.csv(os.path.join(self.path, self.classifier_models[0]), sep=",", header=True, schema=self.get_schema())
                .withColumn("matches", udf_teams(col("team_1"), col("team_2")))
                .withColumnRenamed("prediction", self.classifier_models[0]))
     
        matches_indexer = self.get_indexer().fit(data)

        dic_data[self.classifier_models[0]] = matches_indexer.transform(data)

        for classifier in self.classifier_models[1:]:
            data = (self.spark.read.csv(os.path.join(self.path, classifier), sep=",", header=True, schema=self.get_schema())
                    .withColumn("matches", udf_teams(col("team_1"), col("team_2")))
                    .drop("team_1").drop("team_2").drop("label")
                    .withColumnRenamed("prediction", classifier))     
            dic_data[classifier] = matches_indexer.transform(data).drop("matches")

        return dic_data

    def merge_all_data(self):
        dic_data = self.load_all_data()
        keys = dic_data.keys()
        data = dic_data[keys[0]]
        for key in keys[1:]:
            data = data.join(dic_data[key], on="id")
        data.count()
        return data

    def create_features(self, data):
        if (len(self.classifier_models) == 2):
            udf_features = udf(lambda x,y: Vectors.dense([x, y]), VectorUDT())
            data = data.withColumn("features", udf_features(col(self.classifier_models[0]), col(self.classifier_models[1])))
        elif (len(self.classifier_models) == 3):
            udf_features = udf(lambda x,y,z: Vectors.dense([x, y, z]), VectorUDT())
            data = data.withColumn("features", udf_features(col(self.classifier_models[0]), 
                                                            col(self.classifier_models[1]), 
                                                            col(self.classifier_models[2])))
        elif (len(self.classifier_models) == 4):
            udf_features = udf(lambda t,x,y,z: Vectors.dense([t, x, y, z]), VectorUDT())
            data = data.withColumn("features", udf_features(col(self.classifier_models[0]), 
                                                            col(self.classifier_models[1]), 
                                                            col(self.classifier_models[2]),
                                                            col(self.classifier_models[3])))
        elif (len(self.classifier_models) == 5):
            udf_features = udf(lambda s,t,x,y,z: Vectors.dense([s, t, x, y, z]), VectorUDT())
            data = data.withColumn("features", udf_features(col(self.classifier_models[0]), 
                                                            col(self.classifier_models[1]), 
                                                            col(self.classifier_models[2]),
                                                            col(self.classifier_models[3]),
                                                            col(self.classifier_models[4])))

        for classifier in self.classifier_models: data = data.drop(classifier)

        return data.drop("matches").drop("team_1").drop("team_2")

    def stacking_models(self, data):
        classification_model = ["logistic_regression", "decision_tree", "random_forest", "one_vs_rest"]
#        classification_model = ["multilayer_perceptron"]
        for classifier in classification_model:
            classification_model = ClassificationModel(data, classifier, 
                                                       "./test/stacking", 
                                                       validator="train_validation", 
                                                       list_layers=[[5, 3], [5, 4, 3]])
            print(data.select("features").rdd.map(lambda x: x["features"]).first())
            classification_model.run()
#            print("Classifier: {0} - Accuracy: {1}".format(classifier, classification_model.evaluate_evaluator()))
#            print(classification_model.get_nb_input_layers())
#            print(classification_model.get_nb_output_layers())







