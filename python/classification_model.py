
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
    def __init__(self, data, classification_model, path_model, path_transform, validator=None, list_layers=None):
        self.data = data
        self.model_classifier = classification_model
        self.root_path_model = path_model #"./test/classification_model"
        self.root_path_transform = path_transform
        self.validator= validator
        self.layers = list_layers
        self.featuresCol = "features"
        self.labelCol = "label"
        self.predictionCol = "prediction"

    def __str__(self):
        s = "\nClassification model: {0}\n".format(self.model_classifier)
        s += "Validator: {0}\n".format(self.validator)
        return s

    def run(self):
       self.define_estimator()
       self.define_evaluator()
       self.define_grid_builder()
       self.define_validator()
       self.fit_validator()
       self.save_best_model()
       self.transform_model()
       self.save_transform()

    def get_path_save_model(self):
        return os.path.join(self.root_path_model, self.model_classifier)

    def get_path_transform(self):
        return os.path.join(self.root_path_transform, self.model_classifier)

    def set_model(self, model):
        self.model = model

    def set_transform(self, data):
        self.transform = data

    def define_grid_builder(self):
        if (self.model_classifier =="logistic_regression"):
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.maxIter, [5, 10, 15, 20])
                               .addGrid(self.estimator.regParam, [0.0, 0.01, 0.1, 1.0, 10.0])
                               .addGrid(self.estimator.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
                               .addGrid(self.estimator.fitIntercept, [True, False])
                               .addGrid(self.estimator.aggregationDepth, [2, 4, 8, 16])
                               .build())    
        elif(self.model_classifier == "decision_tree"):
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.maxDepth, [5, 10, 15, 20, 25])
                               .addGrid(self.estimator.maxBins, [4, 8, 16, 32])
                               .build())
        elif(self.model_classifier == "random_forest"):
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.numTrees, [5, 10, 15, 20, 25])
                               .addGrid(self.estimator.maxDepth, [5, 10, 15, 20])
                               .addGrid(self.estimator.maxBins, [4, 8, 16, 32])
                               .build())
        elif(self.model_classifier == "multilayer_perceptron"):
            if self.layers is None:
                self.layers = [[8, 7, 6, 5, 4, 3], [8, 10, 3], [8, 8, 5, 3],  [8, 12, 12, 5, 3]]
            
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.layers, self.layers)
                               .build())
        elif(self.model_classifier == "one_vs_rest"):
            list_classifier = []
            # logistic regression classifier
            regParam = [0.0, 0.01, 0.1, 1.0, 10.0]
            elasticNetParam = [0.0, 0.25, 0.5, 0.75, 1.0]
            for reg,elastic in product(regParam, elasticNetParam):
                list_classifier.append(LogisticRegression(regParam=reg, elasticNetParam=elastic, family="binomial"))
            # linerSVC
            intercept = [True, False]
            for reg, inter in product(regParam, intercept):
                list_classifier.append(LinearSVC(regParam=reg, fitIntercept=inter))            

            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.classifier, list_classifier)
                               .build())

    def define_estimator(self):
        if (self.model_classifier == "logistic_regression"):
            self.estimator = LogisticRegression(featuresCol=self.featuresCol, labelCol=self.labelCol, family="multinomial")
        elif(self.model_classifier == "decision_tree"):
            self.estimator = DecisionTreeClassifier(featuresCol=self.featuresCol, labelCol=self.labelCol)
        elif(self.model_classifier == "random_forest"):
            self.estimator = RandomForestClassifier(featuresCol=self.featuresCol, labelCol=self.labelCol)
        elif(self.model_classifier == "multilayer_perceptron"):
            self.estimator = MultilayerPerceptronClassifier(featuresCol=self.featuresCol, labelCol=self.labelCol)
        elif(self.model_classifier == "one_vs_rest"):
            self.estimator = OneVsRest(featuresCol=self.featuresCol, labelCol=self.labelCol)
    
    def define_evaluator(self):
        self.evaluator = MulticlassClassificationEvaluator(predictionCol=self.predictionCol, labelCol=self.labelCol, metricName="accuracy")

    def define_validator(self):
        if (self.validator == "cross_validation"):
            self.validator = CrossValidator(estimator=self.estimator, 
                                            estimatorParamMaps=self.param_grid, 
                                            evaluator=self.evaluator, 
                                            numFolds=4)
        elif (self.validator == "train_validation"):
#            print(self.get_grid_builder())
            self.validator =  TrainValidationSplit(estimator=self.estimator, 
                                                   estimatorParamMaps=self.param_grid, 
                                                   evaluator=self.evaluator, 
                                                   trainRatio=0.75)

    def fit_validator(self):
        self.model = self.validator.fit(self.data)

    def transform_model(self):
        self.transform = self.model.transform(self.data)

    def evaluate_evaluator(self): 
        return self.evaluator.evaluate(self.transform)

    def save_best_model(self):
        self.model.bestModel.write().overwrite().save(self.get_path_save_model())

    def save_transform(self):
        (self.transform
         .select("id", "label", "prediction")
         .coalesce(1)
         .write.csv(self.get_path_transform(), mode="overwrite", sep=",", header=True))

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

   
