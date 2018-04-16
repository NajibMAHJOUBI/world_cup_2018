from sklearn.linear_model import Lars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from model_definition import DefinitionModel


class RegressionModel(DefinitionModel):

    model_type = "regression"

    def __init__(self, year, model, path_data, path_model):
        DefinitionModel.__init__(self, year, model, self.model_type, path_data, path_model)

        self.estimator = None
        self.param_grid = None
        self.validator = None
        self.data = None

    def __str__(self):
        s = "Year: {0}".format(self.get_year())
        s += "Model: {0}".format(self.get_model())
        s += "Model type: {0}".format(self.get_model_type())
        return s

    def run(self):
        self.load_data()
        self.define_estimator()
        self.define_grid_parameters()
        self.define_validator()
        self.fit_validator()
        self.save_model()

    def define_estimator(self):
        if self.model == "linear_regression":
            self.estimator = LinearRegression()
        elif self.model == "ridge":
            self.estimator = Ridge()
        elif self.model == "lars":
            self.estimator = Lars()

    def define_grid_parameters(self):
        if self.model == "linear_regression":
            self.param_grid = [{'fit_intercept': [True, False],
                                'normalize': [True, False]}]
        elif self.model == "ridge":
            self.param_grid = [{'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                                'fit_intercept': [True, False],
                                'normalize': [True, False],
                                'solver': ['svd', 'cholesky', 'lsqr',
                                           'sparse_cg', 'sag', 'saga']}]
        elif self.model == "lars":
            self.param_grid = [{'fit_intercept': [True, False],
                                'normalize': [True, False],
                                'precompute': [True, False],
                                'positive': [True, False]}]


if __name__ == "__main__":
    from get_regression_models import get_regression_models
    for year in ["2006", "2010", "2014", "2018"]:
        print("Year: {0}".format(year))
        for model in get_regression_models():
            print("  Model: {0}".format(model))
            regression_model = RegressionModel(year, model,
                                               "./test/sklearn/training",
                                               "./test/sklearn/model")
            regression_model.run()
