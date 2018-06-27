import os

import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, \
    GeneralizedLinearRegression, GBTRegressor
from pyspark.ml.regression import LinearRegressionModel, DecisionTreeRegressionModel, RandomForestRegressionModel, \
    GBTRegressionModel, GeneralizedLinearRegressionModel
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType


class RegressionModel:

    def __init__(self, spark, year, model_classifier, path_training, path_model, path_transform, validator):
        self.spark = spark
        self.year = year
        self.model_classifier = model_classifier
        self.path_training = path_training
        self.path_model = path_model
        self.path_transform = path_transform
        self.validator_method = validator

        self.prediction_column = "prediction_diff_points"
        self.label_column = "diff_points"
        self.features_column = "features"
        self.data = None
        self.model = None
        self.transform = None
        self.estimator = None
        self.evaluator = None
        self.validator = None
        self.param_grid = None

    def __str__(self):
        s = "RegressionModel\n"
        s += "model classifier: {0}\n".format(self.model_classifier)
        s += "path training: {0}\n".format(self.path_training)
        s += "path transform: {0}\n".format(self.path_transform)
        s += "path model: {0}\n".format(self.path_model)
        return s

    def run(self):
        self.load_data()
        self.define_estimator()
        self.define_evaluator()
        self.define_grid_builder()
        self.define_validator()
        self.fit_validator()
        self.transform_model()
        self.define_match_issue()
        self.save_best_model()
        self.save_transform()

    def get_transform(self):
        return self.transform

    def get_path_model(self):
        return os.path.join(self.path_model, self.year, self.model_classifier)

    def get_path_transform(self):
        return os.path.join(self.path_transform, self.year, self.model_classifier)

    def get_best_model(self):
        if self.model_classifier == "linear_regression":
            return LinearRegressionModel.load(self.get_path_model())
        elif self.model_classifier == "decision_tree":
            return DecisionTreeRegressionModel.load(self.get_path_model())
        elif self.model_classifier == "random_forest":
            return RandomForestRegressionModel.load(self.get_path_model())
        elif self.model_classifier == "generalized_linear_regression":
            return GeneralizedLinearRegressionModel.load(self.get_path_model())
        elif self.model_classifier == "gbt_regressor":
            return GBTRegressionModel.load(self.get_path_model())

    def get_transform(self):
        return self.transform

    def set_prediction_column(self, prediction):
        self.prediction_column = prediction

    def set_data(self, data):
        self.data = data

    def set_transform(self, transform_to_set):
        self.transform = transform_to_set

    def load_data(self):
        self.data = (self.spark.read.parquet(os.path.join(self.path_training, self.year)))

    def load_best_model(self):
        if self.model_classifier == "linear_regression":
            self.model = LinearRegressionModel.load(self.get_path_model())
        elif self.model_classifier == "decision_tree":
            self.model = DecisionTreeRegressionModel.load(self.get_path_model())
        elif self.model_classifier == "random_forest":
            self.model = RandomForestRegressionModel.load(self.get_path_model())
        elif self.model_classifier == "generalized_linear_regression":
            self.model = GeneralizedLinearRegressionModel.load(self.get_path_model())
        elif self.model_classifier == "gbt_regressor":
            self.model = GBTRegressionModel.load(self.get_path_model())

    def save_best_model(self):
        self.model.bestModel.write().overwrite().save(self.get_path_model())

    def save_transform(self):
        (self.transform
         .select("id", "label", "prediction")
         .coalesce(1)
         .write.csv(self.get_path_transform(), mode="overwrite", sep=",", header=True))

    def define_grid_builder(self):
        if self.model_classifier == "linear_regression":
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.maxIter, [5, 10, 15, 20])
                               .addGrid(self.estimator.regParam, [0.0, 0.01, 0.1, 1.0, 10.0])
                               .addGrid(self.estimator.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
                               .addGrid(self.estimator.fitIntercept, [True, False])
                               .addGrid(self.estimator.aggregationDepth, [2, 4, 8, 16])
                               .build())
        elif self.model_classifier == "decision_tree" or self.model_classifier == "gbt_regressor":
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
        elif self.model_classifier == "generalized_linear_regression":
            self.param_grid = (ParamGridBuilder()
                               .addGrid(self.estimator.regParam, [0.01, 0.1, 1.0, 10.0])
                               .addGrid(self.estimator.family, ["gaussian"])
                               .addGrid(self.estimator.link, ["identity", "log", "inverse"])
                               .addGrid(self.estimator.fitIntercept, [True, False])
                               .build())

    def define_estimator(self):
        if self.model_classifier == "linear_regression":
            self.estimator = LinearRegression()
        elif self.model_classifier == "decision_tree":
            self.estimator = DecisionTreeRegressor()
        elif self.model_classifier == "random_forest":
            self.estimator = RandomForestRegressor()
        elif self.model_classifier == "generalized_linear_regression":
            self.estimator = GeneralizedLinearRegression()
        elif self.model_classifier == "gbt_regressor":
            self.estimator = GBTRegressor()

        self.estimator.setLabelCol(self.label_column)
        self.estimator.setPredictionCol(self.prediction_column)
        self.estimator.setFeaturesCol(self.features_column)

    def define_evaluator(self):
        self.evaluator = RegressionEvaluator(predictionCol=self.prediction_column, labelCol=self.label_column,
                                             metricName="rmse")

    def define_validator(self):
        if self.validator_method == "cross_validation":
            self.validator = CrossValidator(estimator=self.estimator,
                                            estimatorParamMaps=self.param_grid,
                                            evaluator=self.evaluator,
                                            numFolds=4)
        elif self.validator_method == "train_validation":
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

    def define_match_issue(self):
        def get_prediction(x):
            threshold = 0.25
            if float(np.abs(x)) <= threshold:
                return 0.0
            elif x > threshold:
                return 2.0
            elif x < -1.0 * threshold:
                return 1.0
        udf_prediction = udf(lambda x: get_prediction(x), DoubleType())
        self.transform = (self.transform
                          .withColumn("prediction", udf_prediction(col(self.prediction_column)))
                          .drop(self.prediction_column))


if __name__ == "__main__":
    from get_spark_session import get_spark_session
    from classification_model import ClassificationModel
    from get_regression_approach import get_regression_approach
    spark = get_spark_session("Regression Model")
    classification_model = ClassificationModel(None, None, None, None, None, None, None)
    classification_model.define_evaluator()
    regression_models = get_regression_approach()
    dic_year_model = {
        "2018": regression_models,
        # "2014": regression_models,
        # "2010": regression_models,
        # "2006": regression_models
    }
    dic_evaluate_model = {}
    for year, models in dic_year_model.iteritems():
        print("Year: {0}".format(year))
        dic_evaluate_model[year] = {}
        for model in models:
            print("  Model classification: {0}".format(model))
            regression_model = RegressionModel(spark, year, model,
                                               "./src/test/pyspark/training", "./src/test/pyspark/model/regression",
                                               "./src/test/pyspark/transform/regression", "train_validation")
            # spark, year, model_classifier, validator, path_training, path_model, path_transform
            regression_model.run()

            classification_model.set_transform(regression_model.get_transform())
            dic_evaluate_model[year][model] = classification_model.evaluate_evaluator()

    for year in sorted(dic_year_model.keys()):
        print("Year: {0}".format(year))
        print(dic_evaluate_model[year])

