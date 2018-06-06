from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_definition import DefinitionModel


class ClassificationModel(DefinitionModel):

    def __init__(self, year, model, path_data, path_model):

        DefinitionModel.__init__(self, year, model, "classification", path_data, path_model)

        self.estimator = None
        self.param_grid = None
        self.validator = None
        self.data = None

    def __str__(self):
        pass

    def run(self):
        self.load_data()
        self.define_estimator()
        self.define_grid_parameters()
        self.define_validator()
        self.fit_validator()
        self.save_model()

    def define_estimator(self):
        if self.model == "logistic_regression":
            self.estimator = LogisticRegression(max_iter=200, tol=0.001)
        elif self.model == "k_neighbors":
            self.estimator = KNeighborsClassifier()
        elif self.model == "svc":
            self.estimator = SVC()
        elif self.model == "gaussian_classifier":
            self.estimator = GaussianProcessClassifier()
        elif self.model == "decision_tree":
            self.estimator = DecisionTreeClassifier()
        elif self.model == "random_forest":
            self.estimator = RandomForestClassifier()
        elif self.model == "mlp_classifier":
            self.estimator = MLPClassifier()
        elif self.model == "ada_boost":
            self.estimator = AdaBoostClassifier()
        elif self.model == "gaussian":
            self.estimator = GaussianNB()
        elif self.model == "quadratic":
            self.estimator = QuadraticDiscriminantAnalysis()
        elif self.model == "sgd":
            self.estimator = SGDClassifier()

    def define_grid_parameters(self):
        if self.model == 'logistic_regression':
            self.param_grid = [{'penalty': ['l2'],
                                'C': [1, 10, 100, 1000],
                                'solver': ['newton-cg', 'sag', 'lbfgs'],
                                'fit_intercept': [True, False]},
                               {'penalty': ['l1'],
                                'C': [4.0, 2.0, 1.333, 1.0],
                                'solver': ['liblinear', 'saga'],
                                'fit_intercept': [True, False]}]
        elif self.model == "k_neighbors":
            self.param_grid = [{'n_neighbors': [2, 5, 10, 15, 20],
                                'algorithm': ['brute']},
                               {'n_neighbors': [2, 5, 10, 15, 20],
                                'algorithm': ['ball_tree', 'kd_tree'],
                                'leaf_size': [10, 20, 30, 40]}]
        elif self.model == "svc":
            self.param_grid = [{'kernel': ['rbf', 'linear', 'sigmoid'],
                                'C': [1, 10, 100, 1000],
                                'gamma': [0.001, 0.0001, 1./8]},
                               {'kernel': ['poly'],
                                'C': [1, 10, 100, 1000],
                                'gamma': [0.001, 0.0001, 1./8],
                                'degree': [2, 3, 4, 5]}]
        elif self.model == "gaussian_classifier":
            self.param_grid = [{'kernel': [1.0 * RBF(1.0),
                                           1.0 * RBF(1000.0),
                                           1.0 * RBF(0.001)]}]
        elif self.model == "decision_tree":
            self.param_grid = [{'criterion': ['gini', 'entropy'],
                                'max_depth': [5, 10, 15, 20, 25, None],
                                'min_samples_split': [2, 4, 6, 8],
                                'min_samples_leaf': [1, 2, 4, 8],
                                'max_features': ['sqrt', 'log2', None]}]
        elif self.model == "random_forest":
            self.param_grid = [{'n_estimators': [5, 10, 20],
                                'criterion': ['gini', 'entropy'],
                                'max_depth': [5, 10, 15, 20, 25, None],
                                'min_samples_split': [2, 4, 6, 8],
                                'min_samples_leaf': [1, 2, 4, 8],
                                'max_features': ['sqrt', 'log2', None]}]
        elif self.model == "mlp_classifier":
            self.param_grid = [{'hidden_layer_sizes': [[8, 7, 6, 5, 4, 3],
                                                       [8, 10, 3],
                                                       [8, 8, 5, 3],
                                                       [8, 12, 12, 5, 3]],
                                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                'solver': ['lbfgs', 'sgd', 'adam'],
                                'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0],
                                'learning_rate': ['constant', 'invscaling', 'adaptive']}]
        elif self.model == "ada_boost":
            self.param_grid = [{'base_estimator': [LogisticRegression(),
                                                   DecisionTreeClassifier(),
                                                   RandomForestClassifier()],
                                'algorithm': ['SAMME', 'SAMME.R']}]
        elif self.model == "gaussian":
            self.param_grid = [{}
                               ]
        elif self.model == "quadratic":
            self.param_grid = [{'reg_param': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]}
                               ]
        elif self.model == "sgd":
            self.param_grid = [{'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                                         'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                                'penalty': ["elasticnet"],
                                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                                'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
                                'fit_intercept': [True, False]}]


if __name__ == "__main__":
    import sys
    sys.path.append("./pyspark")
    from get_classification_models import get_classification_models
    for year in ["2006", "2010", "2014", "2018"]:
        print("Year: {0}".format(year))
        for model in get_classification_models(["gaussian_classifier"]):
            print("  Model: {0}".format(model))
            classification_model = ClassificationModel(year, model,
                                                       "./test/sklearn/training",
                                                       "./test/sklearn/model")
            classification_model.run()
