
# pyspark libraries
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, MultilayerPerceptronClassifier, OneVsRest, LinearSVC, GBTClassifier
from pyspark.ml.classification import LogisticRegressionModel, DecisionTreeClassificationModel, RandomForestClassificationModel,  MultilayerPerceptronClassificationModel, OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
# python libraries
import os

# Class Classification Model
class ClassificationModel:
    def __init__(self, data, classification_model, validator=None):
        self.data = data
        self.model_classifier = classification_model
        self.validator= validator
        self.featuresCol = "features"
        self.labelCol = "label"
        self.predictionCol = "prediction"
        self.root_path_model = "./test/classification_model"
 
    def __str__(self):
        s = "Classification model: {0}".format(self.model)
        return s

    def run(self):
       self.get_estimator()
       self.param_grid_builder()
       
       self.get_evaluator()
       self.get_validator()
       self.fit_validator()
       self.save_best_model()
       #self.evaluate_evaluator()

    def get_path_save_model(self):
        return os.path.join(self.root_path_model ,self.model_classifier)

    def param_grid_builder(self):
        if (self.model_classifier =="logistic_regression"):
            self.grid = ParamGridBuilder()\
                       .addGrid(self.estimator.maxIter, [5, 10, 15, 20])\
                       .addGrid(self.estimator.regParam, [0.0, 0.01, 0.1, 1.0, 10.0])\
                       .addGrid(self.estimator.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])\
                       .addGrid(self.estimator.fitIntercept, [True, False])\
                       .addGrid(self.estimator.aggregationDepth, [2, 4, 8, 16])\
                       .build()    
        elif(self.model_classifier == "decision_tree"):
            self.grid = ParamGridBuilder()\
                        .addGrid(self.estimator.maxDepth, [5, 10, 15, 20, 25])\
                        .addGrid(self.estimator.maxBins, [4, 8, 16, 32])\
                        .build()
        elif(self.model_classifier == "random_forest"):
            self.grid = ParamGridBuilder()\
                        .addGrid(self.estimator.numTrees, [5, 10, 15, 20, 25])\
                        .addGrid(self.estimator.maxDepth, [5, 10, 15, 20])\
                        .addGrid(self.estimator.maxBins, [4, 8, 16, 32])\
                        .build()
        elif(self.model_classifier == "multilayer_perceptron"):
            self.grid = ParamGridBuilder()\
                        .addGrid(self.estimator.layers, [[8, 7, 6, 5, 4, 3], 
                                                         [8, 10, 3], [8, 8, 5, 3],
                                                         [8, 12, 12, 5, 3]])\
                        .build()            
        elif(self.model_classifier == "one_vs_rest"):
            lr_0 = LogisticRegression(regParam=0.0, elasticNetParam=0.0, family="binomial")
            lr_1 = LogisticRegression(regParam=0.5, elasticNetParam=0.0, family="binomial")
            lr_2 = LogisticRegression(regParam=1.0, elasticNetParam=0.0, family="binomial")
            lr_3 = LogisticRegression(regParam=0.0, elasticNetParam=0.5, family="binomial")
            lr_4 = LogisticRegression(regParam=0.5, elasticNetParam=0.5, family="binomial")
            lr_5 = LogisticRegression(regParam=1.0, elasticNetParam=0.5, family="binomial")
            lr_6 = LogisticRegression(regParam=0.0, elasticNetParam=1.0, family="binomial")
            lr_7 = LogisticRegression(regParam=0.5, elasticNetParam=1.0, family="binomial")
            lr_8 = LogisticRegression(regParam=1.0, elasticNetParam=1.0, family="binomial")
            svc_0 = LinearSVC(regParam=0.0, fitIntercept=True)
            svc_1 = LinearSVC(regParam=0.5, fitIntercept=True)
            svc_2 = LinearSVC(regParam=1.0, fitIntercept=True)
            svc_3 = LinearSVC(regParam=0.0, fitIntercept=False)
            svc_4 = LinearSVC(regParam=0.5, fitIntercept=False)
            svc_5 = LinearSVC(regParam=1.0, fitIntercept=False)
            gb_0 = GBTClassifier(maxDepth=10, maxBins=16)
            gb_1 = GBTClassifier(maxDepth=10, maxBins=16)
            gb_2 = GBTClassifier(maxDepth=10, maxBins=16)
            gb_3 = GBTClassifier(maxDepth=20, maxBins=32)
            gb_4 = GBTClassifier(maxDepth=20, maxBins=32)
            gb_5 = GBTClassifier(maxDepth=20, maxBins=32)

            self.grid = ParamGridBuilder()\
                        .addGrid(self.estimator.classifier, [lr_0, lr_1, lr_2, lr_3, lr_4, lr_5, lr_6, lr_7, lr_8, 
                                                             svc_0, svc_1, svc_2, svc_3, svc_4, svc_5,
                                                             gb_0, gb_1, gb_2, gb_3, gb_4, gb_5])\
                        .build()              

    def get_estimator(self):
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
    
    def get_evaluator(self):
        return MulticlassClassificationEvaluator(predictionCol=self.predictionCol, labelCol=self.labelCol, metricName="accuracy")

    def get_validator(self):
        if (self.validator == "cross_validation"):
            self.validation = CrossValidator(estimator=self.estimator, estimatorParamMaps=self.grid, evaluator=self.get_evaluator(), numFolds=4)
        elif (self.validator == "train_validation"):
            self.validation = TrainValidationSplit(estimator=self.estimator, estimatorParamMaps=self.grid, evaluator=self.get_evaluator(), trainRatio=0.75)
        else:
            self.train, self.test = self.data.randomSplit([0.8, 0.2])

    def fit_validator(self):
        if (self.validator is None):
            self.model = self.estimator.setMaxIter(20).setRegParam(0.0).fit(self.train)
        else:
            self.model = self.validation.fit(self.data)

    def transform_model(self, data):
        return self.model.transform(data)

    def evaluate_evaluator(self):
        if (self.validator is None):
            train_prediction = self.transform_model(self.train)
            test_prediction = self.transform(self.test)
            print("Accuracy on the train dataset: {0}".format(self.get_evaluator().evaluate(train_prediction)))
            print("Accuracy on the test dataset: {0}".format(self.get_evaluator().evaluate(test_prediction)))
        else:
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
        


   
