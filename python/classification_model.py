
# pyspark libraries
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, MultilayerPerceptronClassifier, OneVsRest, LinearSVC, GBTClassifier
from pyspark.ml.classification import LogisticRegressionModel, DecisionTreeClassificationModel, RandomForestClassificationModel,  MultilayerPerceptronClassificationModel, OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
# python libraries
import os
from itertools import product
import unittest

# Class Classification Model
class ClassificationModel(unittest.TestCase):
    def __init__(self, data, classification_model, path_model, validator=None, list_layers=None):
        self.data = data
        self.model_classifier = classification_model
        self.validator= validator
        self.layers = list_layers
        self.featuresCol = "features"
        self.labelCol = "label"
        self.predictionCol = "prediction"
        self.root_path_model = path_model #"./test/classification_model"
 
    def __str__(self):
        s = "\nClassification model: {0}\n".format(self.model_classifier)
        s += "Validator: {0}\n".format(self.validator)
        return s

    def run(self):
       self.fit_validator()
#       self.save_best_model()
       #self.evaluate_evaluator()

    def get_path_save_model(self):
        return os.path.join(self.root_path_model ,self.model_classifier)

    def get_grid_builder(self):
        if (self.model_classifier =="logistic_regression"):
            return ParamGridBuilder()\
                       .addGrid(self.get_estimator().maxIter, [5, 10, 15, 20])\
                       .addGrid(self.get_estimator().regParam, [0.0, 0.01, 0.1, 1.0, 10.0])\
                       .addGrid(self.get_estimator().elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])\
                       .addGrid(self.get_estimator().fitIntercept, [True, False])\
                       .addGrid(self.get_estimator().aggregationDepth, [2, 4, 8, 16])\
                       .build()    
        elif(self.model_classifier == "decision_tree"):
            return ParamGridBuilder()\
                        .addGrid(self.get_estimator().maxDepth, [5, 10, 15, 20, 25])\
                        .addGrid(self.get_estimator().maxBins, [4, 8, 16, 32])\
                        .build()
        elif(self.model_classifier == "random_forest"):
            return ParamGridBuilder()\
                        .addGrid(self.get_estimator().numTrees, [5, 10, 15, 20, 25])\
                        .addGrid(self.get_estimator().maxDepth, [5, 10, 15, 20])\
                        .addGrid(self.get_estimator().maxBins, [4, 8, 16, 32])\
                        .build()
        elif(self.model_classifier == "multilayer_perceptron"):
            return (ParamGridBuilder()
                    .addGrid(self.get_estimator().layers, [[8, 7, 6, 5, 4, 3], [8, 10, 3], [8, 8, 5, 3], [8, 12, 12, 5, 3]])
                    .build())
        elif(self.model_classifier == "one_vs_rest"):
            list_classifier = []
            # logistic regression classifier
            regParam = [0.0, 0.5, 1.0]
            elasticNetParam = [0.0, 0.25, 0.5, 0.75, 1.0]
            for reg,elastic in product(regParam, elasticNetParam):
                list_classifier.append(LogisticRegression(regParam=reg, elasticNetParam=elastic, family="binomial"))
            # linerSVC
            intercept = [True, False]
            for reg, inter in product(regParam, intercept):
                list_classifier.append(LinearSVC(regParam=reg, fitIntercept=inter))            

            return ParamGridBuilder()\
                        .addGrid(get_estimator().classifier, list_classifier)\
                        .build()              

    def get_estimator(self):
        if (self.model_classifier == "logistic_regression"):
            return LogisticRegression(featuresCol=self.featuresCol, labelCol=self.labelCol, family="multinomial")
        elif(self.model_classifier == "decision_tree"):
            return DecisionTreeClassifier(featuresCol=self.featuresCol, labelCol=self.labelCol)
        elif(self.model_classifier == "random_forest"):
            return RandomForestClassifier(featuresCol=self.featuresCol, labelCol=self.labelCol)
        elif(self.model_classifier == "multilayer_perceptron"):
            return MultilayerPerceptronClassifier(featuresCol=self.featuresCol, labelCol=self.labelCol)
        elif(self.model_classifier == "one_vs_rest"):
            return OneVsRest(featuresCol=self.featuresCol, labelCol=self.labelCol)
    
    def get_evaluator(self):
        return MulticlassClassificationEvaluator(predictionCol=self.predictionCol, labelCol=self.labelCol, metricName="accuracy")

    def get_validator(self):
        if (self.validator == "cross_validation"):
            return CrossValidator(estimator=self.get_estimator(), 
                                  estimatorParamMaps=self.get_grid_builder(), 
                                  evaluator=self.get_evaluator(), 
                                  numFolds=4)
        elif (self.validator == "train_validation"):
#            print(self.get_grid_builder())
            return TrainValidationSplit(estimator=self.get_estimator(), 
                                        estimatorParamMaps=self.get_grid_builder(), 
                                        evaluator=self.get_evaluator(), 
                                        trainRatio=0.75)

    def fit_validator(self):
#        print(self.get_validator().getEstimatorParamMaps())

        self.model = self.get_validator().fit(self.data)

    def transform_model(self, data):
        return self.model.transform(data)

    def evaluate_evaluator(self):
        prediction = self.transform_model(self.data)      
        return self.get_evaluator().evaluate(prediction)

    def save_best_model(self):
        self.model.bestModel.write().overwrite().save(self.get_path_save_model())

    def get_best_model(self):
        if (self.model_classifier == "logistic_regression"):
            return LogisticRegressionModel.load(self.get_path_save_model())
        elif(self.model_classifier == "decision_tree"):
            return DecisionTreeClassificationModel.load(self.get_path_save_model())
        elif(self.model_classifier == "random_forest"):
            return RandomForestClassificationModel.load(self.get_path_save_model())
        elif(self.model_classifier == "multilayer_perceptron"):
            return MultilayerPerceptronClassificationModel.load(self.get_path_save_model())
        elif(self.model_classifier == "one_vs_rest"):
            return OneVsRestModel.load(self.get_path_save_model())
        
    def get_nb_input_layers(self):
        return self.data.select(self.featuresCol).rdd.map(lambda x: x[self.featuresCol]).first().values.size

    def get_nb_output_layers(self):
        return self.data.select(self.labelCol).distinct().count()

   
