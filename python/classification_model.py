
import os
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from get_features import get_features

class ClassificationModel:

    def __init__(self, year, model_classifier, path_data, path_model):
        self.year = year
        self.model_classifier = model_classifier
        self.path_data = path_data
        self.path_model = path_model

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

    def get_path_data(self):
        return os.path.join(self.path_data, self.year+".csv")

    def get_path_model(self):
        return os.path.join(self.path_model, self.year, self.model_classifier+".pkl")

    def get_x(self):
        return self.data.loc[:, get_features("features")]

    def get_y(self):
        label = get_features("label")
        return self.data.label

    def load_data(self):
        self.data = pd.read_csv(self.get_path_data(), header=0, sep=",")

    def define_estimator(self):
        if self.model_classifier == "logistic_regression":
            self.estimator = LogisticRegression(max_iter=200, tol=0.001)
        elif self.model_classifier == "k_neighbors":
            self.estimator = KNeighborsClassifier()
        elif self.model_classifier == "svc":
            self.estimator = SVC()
        elif self.model_classifier == "gaussian_classifier":
            self.estimator = GaussianProcessClassifier()
        elif self.model_classifier == "decision_tree":
            self.estimator = DecisionTreeClassifier()
        elif self.model_classifier == "random_forest":
            self.estimator = RandomForestClassifier()
        elif self.model_classifier == "mlp_classifier":
            self.estimator = MLPClassifier()
        elif self.model_classifier == "ada_boost":
            self.estimator = AdaBoostClassifier()
        elif self.model_classifier == "gaussian":
            self.estimator = GaussianNB()
        elif self.model_classifier == "quadratic":
            self.estimator = QuadraticDiscriminantAnalysis()
        elif self.model_classifier == "sgd":
            self.estimator = SGDClassifier()

    def define_grid_parameters(self):
        if self.model_classifier == 'logistic_regression':
            self.param_grid = [{'penalty': ['l2'],
                                'C': [1, 10, 100, 1000],
                                'solver': ['newton-cg', 'sag', 'lbfgs'],
                                'fit_intercept': [True, False]},
                               {'penalty': ['l1'],
                                'C': [4.0, 2.0, 1.333, 1.0],
                                'solver': ['liblinear', 'saga'],
                                'fit_intercept': [True, False]}]
        elif self.model_classifier == "k_neighbors":
            self.param_grid = [{'n_neighbors': [2, 5, 10, 15, 20],
                                'algorithm': ['brute']},
                               {'n_neighbors': [2, 5, 10, 15, 20],
                                'algorithm': ['ball_tree', 'kd_tree'],
                                'leaf_size': [10, 20, 30, 40]}]
        elif self.model_classifier == "svc":
            self.param_grid = [{'kernel': ['rbf', 'linear', 'sigmoid'],
                                'C': [1, 10, 100, 1000],
                                'gamma': [0.001, 0.0001, 1./8]},
                               {'kernel': ['poly'],
                                'C': [1, 10, 100, 1000],
                                'gamma': [0.001, 0.0001, 1./8],
                                'degree': [2, 3, 4, 5]}]
        elif self.model_classifier == "gaussian_classifier":
            self.param_grid = [{'kernel': [1.0 * RBF(1.0),
                                           1.0 * RBF(1000.0),
                                           1.0 * RBF(0.001)]}]
        elif self.model_classifier == "decision_tree":
            self.param_grid = [{'criterion': ['gini', 'entropy'],
                                'max_depth': [5, 10, 15, 20, 25, None],
                                'min_samples_split': [2, 4, 6, 8],
                                'min_samples_leaf': [1, 2, 4, 8],
                                'max_features': ['sqrt', 'log2', None]}]
        elif self.model_classifier == "random_forest":
            self.param_grid = [{'n_estimators': [5, 10, 20],
                                'criterion': ['gini', 'entropy'],
                                'max_depth': [5, 10, 15, 20, 25, None],
                                'min_samples_split': [2, 4, 6, 8],
                                'min_samples_leaf': [1, 2, 4, 8],
                                'max_features': ['sqrt', 'log2', None]}]
        elif self.model_classifier == "mlp_classifier":
            self.param_grid = [{'hidden_layer_sizes': [[8, 7, 6, 5, 4, 3],
                                                       [8, 10, 3],
                                                       [8, 8, 5, 3],
                                                       [8, 12, 12, 5, 3]],
                                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                'solver': ['lbfgs', 'sgd', 'adam'],
                                'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0],
                                'learning_rate': ['constant', 'invscaling', 'adaptive']}]
        elif self.model_classifier == "ada_boost":
            self.param_grid = [{'base_estimator': [LogisticRegression(),
                                                   DecisionTreeClassifier(),
                                                   RandomForestClassifier()],
                                'algorithm': ['SAMME', 'SAMME.R']}]
        elif self.model_classifier == "gaussian":
            self.param_grid = [{}
                               ]
        elif self.model_classifier == "quadratic":
            self.param_grid = [{'reg_param': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]}
                               ]
        elif self.model_classifier == "sgd":
            self.param_grid = [{'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss',
                                         'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                                'penalty': ["elasticnet"],
                                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                                'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
                                'fit_intercept': [True, False]}]

    def define_validator(self):
        self.validator = GridSearchCV(self.estimator, self.param_grid, cv=4, scoring='accuracy')

    def fit_validator(self):
        self.validator.fit(self.get_x(), self.get_y())

    def transform_model(self):
        return self.validator.predict(self.get_x())

    def save_model(self):
        if not os.path.isdir(os.path.join(self.path_model, self.year)):
            os.makedirs(os.path.join(self.path_model, self.year))
        if os.path.isfile(self.get_path_model()):
            os.remove(self.get_path_model())
        joblib.dump(self.validator, self.get_path_model())

    def load_model(self):
        self.validator = joblib.load(self.get_path_model())

    def evaluate(self):
        return accuracy_score(self.get_y(), self.transform_model(), normalize=True)


if __name__ == "__main__":
    # models = ["logistic_regression", "k_neighbors", "gaussian_classifier",
    #           "decision_tree", "random_forest", "mlp_classifier", "ada_boost",
    #           "gaussian", "quadratic", "svc"]
    models = ["svc"]
    dic_accuracy = {}
    for model in models:
        print("Model: {0}".format(model))
        classification_model = ClassificationModel("2014", model,
                                                   "./test/sklearn/training",
                                                   "./test/sklearn/model")
        classification_model.run()
        dic_accuracy[model] = classification_model.evaluate()

for item in dic_accuracy.iteritems():
    print item
