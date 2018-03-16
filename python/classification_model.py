
# Import libraries
import os
from itertools import product

from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, \
    MultilayerPerceptronClassifier, OneVsRest, LinearSVC
from pyspark.ml.classification import LogisticRegressionModel, DecisionTreeClassificationModel, \
    RandomForestClassificationModel, MultilayerPerceptronClassificationModel, OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder

from get_spark_session import get_spark_session


# Class Classification Model
class ClassificationModel:
    def __init__(self, spark_session, year, classification_model, path_data, path_model, path_transform,
                 validator, list_layers=None):
        self.spark = spark_session
        self.year = year
        self.model_classifier = classification_model
        self.root_path_data = path_data
        self.root_path_model = path_model
        self.root_path_transform = path_transform
        self.validator = validator
        self.layers = list_layers
        self.features_column = "features"
        self.label_column = "label"
        self.prediction_column = "prediction"

        self.data = None  # data - training
        self.transform = None  # prediction
        self.model = None  # classifier model
        self.param_grid = None  # parameters grid builder
        self.estimator = None  # model estimator
        self.evaluator = None  # evaluator multi label prediction

    def __str__(self):
        s = "\nClassification model: {0}\n".format(self.model_classifier)
        s += "Validator: {0}\n".format(self.validator)
        return s

    def run(self):
        self.load_data()
        self.define_estimator()
        self.define_evaluator()
        self.define_grid_builder()
        self.define_validator()
        self.fit_validator()
        self.save_best_model()
        self.transform_model()
        self.save_transform()

    def get_path_save_model(self):
        return os.path.join(self.root_path_model, self.year, self.model_classifier)

    def get_path_transform(self):
        return os.path.join(self.root_path_transform, self.year, self.model_classifier)

    def get_best_model(self):
        if self.model_classifier == "logistic_regression":
            return LogisticRegressionModel.load(self.get_path_save_model())
        elif self.model_classifier == "decision_tree":
            return DecisionTreeClassificationModel.load(self.get_path_save_model())
        elif self.model_classifier == "random_forest":
            return RandomForestClassificationModel.load(self.get_path_save_model())
        elif self.model_classifier == "multilayer_perceptron":
            return MultilayerPerceptronClassificationModel.load(self.get_path_save_model())
        elif self.model_classifier == "one_vs_rest":
            return OneVsRestModel.load(self.get_path_save_model())

    def get_transform(self):
        return self.transform

    def set_model(self, model_to_set):
        self.model = model_to_set

    def set_transform(self, transform_to_set):
        self.transform = transform_to_set

    def set_data(self, data_to_set):
        self.data = data_to_set

    def load_data(self):
        self.data = (self.spark.read.parquet(os.path.join(self.root_path_data, self.year)))

    def load_best_model(self):
        if self.model_classifier == "logistic_regression":
            self.model = LogisticRegressionModel.load(self.get_path_save_model())
        elif self.model_classifier == "decision_tree":
            self.model = DecisionTreeClassificationModel.load(self.get_path_save_model())
        elif self.model_classifier == "random_forest":
            self.model = RandomForestClassificationModel.load(self.get_path_save_model())
        elif self.model_classifier == "multilayer_perceptron":
            self.model = MultilayerPerceptronClassificationModel.load(self.get_path_save_model())
        elif self.model_classifier == "one_vs_rest":
            self.model = OneVsRestModel.load(self.get_path_save_model())

    def define_grid_builder(self):
        if self.model_classifier == "logistic_regression":
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.maxIter, [5, 10, 15, 20])
                               .addGrid(self.estimator.regParam, [0.0, 0.01, 0.1, 1.0, 10.0])
                               .addGrid(self.estimator.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
                               .addGrid(self.estimator.fitIntercept, [True, False])
                               .addGrid(self.estimator.aggregationDepth, [2, 4, 8, 16])
                               .build())
        elif self.model_classifier == "decision_tree":
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.maxDepth, [5, 10, 15, 20, 25])
                               .addGrid(self.estimator.maxBins, [4, 8, 16, 32])
                               .build())
        elif self.model_classifier == "random_forest":
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.numTrees, [5, 10, 15, 20, 25])
                               .addGrid(self.estimator.maxDepth, [5, 10, 15, 20])
                               .addGrid(self.estimator.maxBins, [4, 8, 16, 32])
                               .build())
        elif self.model_classifier == "multilayer_perceptron":
            if self.layers is None:
                self.layers = [[8, 7, 6, 5, 4, 3], [8, 10, 3], [8, 8, 5, 3],  [8, 12, 12, 5, 3]]

            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.layers, self.layers)
                               .build())
        elif self.model_classifier == "one_vs_rest":
            list_classifier = []
            # logistic regression classifier
            reg_param = [0.0, 0.01, 0.1, 1.0, 10.0]
            elastic_param = [0.0, 0.25, 0.5, 0.75, 1.0]
            for reg, elastic in product(reg_param, elastic_param):
                list_classifier.append(LogisticRegression(regParam=reg, elasticNetParam=elastic, family="binomial"))
            # linerSVC
            intercept = [True, False]
            for reg, inter in product(reg_param, intercept):
                list_classifier.append(LinearSVC(regParam=reg, fitIntercept=inter))

            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.classifier, list_classifier)
                               .build())

    def define_estimator(self):
        if self.model_classifier == "logistic_regression":
            self.estimator = LogisticRegression(featuresCol=self.features_column, labelCol=self.label_column,
                                                family="multinomial")
        elif self.model_classifier == "decision_tree":
            self.estimator = DecisionTreeClassifier(featuresCol=self.features_column, labelCol=self.label_column)
        elif self.model_classifier == "random_forest":
            self.estimator = RandomForestClassifier(featuresCol=self.features_column, labelCol=self.label_column)
        elif self.model_classifier == "multilayer_perceptron":
            self.estimator = MultilayerPerceptronClassifier(featuresCol=self.features_column,
                                                            labelCol=self.label_column)
        elif self.model_classifier == "one_vs_rest":
            self.estimator = OneVsRest(featuresCol=self.features_column, labelCol=self.label_column)

    def define_evaluator(self):
        self.evaluator = MulticlassClassificationEvaluator(predictionCol=self.prediction_column,
                                                           labelCol=self.label_column,
                                                           metricName="accuracy")

    def define_validator(self):
        if self.validator == "cross_validation":
            self.validator = CrossValidator(estimator=self.estimator, 
                                            estimatorParamMaps=self.param_grid, 
                                            evaluator=self.evaluator, 
                                            numFolds=4)
        elif self.validator == "train_validation":
            self.validator = TrainValidationSplit(estimator=self.estimator,
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


if __name__ == "__main__":
    spark = get_spark_session("Classification Model")
    classification_models = ["logistic_regression", "decision_tree", "random_forest", "multilayer_perceptron",
                            "one_vs_rest"]
    dic_year_model = {
        "2018": classification_models,
        "2014": classification_models,
        "2010": classification_models,
        "2006": classification_models,
    }

    dic_evaluate_model = {}
    # classification_models = ["one_vs_rest"]
    for year, models in dic_year_model.iteritems():
        print("Year: {0}".format(year))
        dic_evaluate_model[year] = {}
        for model in models:
            print("  Model classification: {0}".format(model))
            classification_model = ClassificationModel(spark, year, model,
                                                       "./test/training",
                                                       "./test/model/classification_model",
                                                       "./test/transform/classification_model",
                                                       validator="train_validation", list_layers=None)
            classification_model.run()
            dic_evaluate_model[year][model] = classification_model.evaluate_evaluator()

    for year in sorted(dic_year_model.keys()):
        print(dic_evaluate_model[year])
