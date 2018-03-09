
import os
import itertools

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructField, StructType, StringType, FloatType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT

from classification_model import ClassificationModel

class Stacking:
    def __init__(self, spark_session, available_classifier, stacking_classifier, validator, path_data):
        self.spark = spark_session
        self.classifier_available = available_classifier
        self.classifier_stacking = stacking_classifier # "decision_tree", "logistic_regression", "decision_tree", "random_forest", "one_vs_rest"
        self.validator = validator
        self.path = path_data # "./test/transform"

        self.data = None
        self.stacking_model = None

    def __str__(self):
        s = "Stacking model\n"
        s += "Available classifier: {0}".format(self.classifier_available)
        s += "Stacking classifier: {0}".format(self.stacking_classifier)
        return s

    def run(self):
        self.merge_all_data()
        self.create_features(self.data)
        self.stacking_models(self.new_data)
        self.save_best_model()

    def get_schema(self):
        return (StructType([
                StructField("id", FloatType(), True),
	            StructField("label", FloatType(), True),
                StructField("prediction", FloatType(), True)]))

    def load_all_data(self):
        dic_data = {
            self.classifier_available[0]:  (self.spark.read
                                            .csv(os.path.join(self.path, self.classifier_available[0]),
                                                 sep=",", header=True, schema=self.get_schema())
                                            .withColumnRenamed("prediction", self.classifier_available[0]))
        }


        for classifier in self.classifier_available[1:]:
            dic_data[classifier] = (self.spark.read
                                    .csv(os.path.join(self.path, classifier), sep=",", header=True, schema=self.get_schema())
                                    .drop("label")
                                    .withColumnRenamed("prediction", classifier))
        return dic_data

    def merge_all_data(self):
        dic_data = self.load_all_data()
        keys = dic_data.keys()
        self.data = dic_data[keys[0]]
        for key in keys[1:]:
            self.data = self.data.join(dic_data[key], on="id")
        self.data.count()

    def create_features(self, data):
        if len(self.classifier_available) == 2:
            udf_features = udf(lambda x,y: Vectors.dense([x, y]), VectorUDT())
            self.new_data = data.withColumn("features", udf_features(col(self.classifier_models[0]), col(self.classifier_models[1])))
        elif len(self.classifier_available) == 3:
            udf_features = udf(lambda x,y,z: Vectors.dense([x, y, z]), VectorUDT())
            self.new_data = data.withColumn("features", udf_features(col(self.classifier_available[0]), 
                                                            col(self.classifier_available[1]), 
                                                            col(self.classifier_available[2])))
        elif len(self.classifier_available) == 4:
            udf_features = udf(lambda t,x,y,z: Vectors.dense([t, x, y, z]), VectorUDT())
            self.new_data = data.withColumn("features", udf_features(col(self.classifier_available[0]), 
                                                            col(self.classifier_available[1]), 
                                                            col(self.classifier_available[2]),
                                                            col(self.classifier_available[3])))
        elif len(self.classifier_available) == 5:
            udf_features = udf(lambda s,t,x,y,z: Vectors.dense([s, t, x, y, z]), VectorUDT())
            self.new_data = data.withColumn("features", udf_features(col(self.classifier_available[0]), 
                                                            col(self.classifier_available[1]), 
                                                            col(self.classifier_available[2]),
                                                            col(self.classifier_available[3]),
                                                            col(self.classifier_available[4])))

        for classifier in self.classifier_available: self.new_data = self.new_data.drop(classifier)
        self.new_data.drop("matches").drop("team_1").drop("team_2")

    def stacking_models(self, data):
        classification_model = ClassificationModel(data, self.classifier_stacking, 
                                                   "./test/classification_model/stacking", 
                                                   "./test/transform/stacking",
                                                   validator="train_validation", 
                                                   list_layers=[[5, 3], [5, 4, 3], [5, 7, 7,5,7,5,3]])
        classification_model.run()
        self.stacking_model = classification_model.get_best_model()

    def save_best_model(self):
        self.stacking_model.write().overwrite().save("./test/classification_model/stacking/{0}".format(self.classifier_stacking))

    def stacking_transform(self):
        return self.stacking_model.transform(self.new_data) 

