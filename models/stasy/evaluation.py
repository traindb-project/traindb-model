import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from tqdm import tqdm

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

_MODELS = {
    'binary_classification': [ 
         {
             'class': DecisionTreeClassifier, 
             'kwargs': {
                 'max_depth': [4, 8, 16, 32], 
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 3, 5]
             }
         },
         {
             'class': AdaBoostClassifier, 
             'kwargs': {
                 'n_estimators': [10, 50, 100]
             }
         },
         {
            'class': LogisticRegression,
            'kwargs': {
                 'solver': ['lbfgs'],
                 'n_jobs': [-1],
                 'max_iter': [10, 50, 100],
                 'C': [0.01, 0.1, 1.0],
                 'tol': [1e-01, 1e-02, 1e-04]
             }
         },
        {
            'class': MLPClassifier, 
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
        {
            'class': RandomForestClassifier, 
            'kwargs': {
                 'max_depth': [8, 16, None], 
                 'min_samples_split': [2, 4],
                 'min_samples_leaf': [1, 3],
                'n_jobs': [-1]

            }
        },
        {
            'class': XGBClassifier,
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10], 
                 'max_depth': [5, 10],
                 'gamma': [0.0, 1.0],
                 'objective': ['binary:logistic'],
                 'nthread': [-1]
            },
        }



    ],
    'multiclass_classification': [ 
        {
            'class': XGBClassifier, 
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10], 
                 'max_depth': [5, 10],
                 'gamma': [0.0, 1.0],
                 'objective': ['binary:logistic'],
                 'nthread': [-1]
            }
        },

        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
         {
             'class': DecisionTreeClassifier,
             'kwargs': {
                 'max_depth': [4, 8, 16, 32], 
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 3, 5]
             }
         },
        {
            'class': RandomForestClassifier, 
            'kwargs': {
                 'max_depth': [8, 16, None], 
                 'min_samples_split': [2, 4],
                 'min_samples_leaf': [1, 3],
                'n_jobs': [-1]

            }
        }

    ],
    'regression': [ # 48
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor, 
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
        {
            'class': XGBRegressor,
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10], 
                 'max_depth': [5, 10],
                 'gamma': [0.0, 1.0],
                 'objective': ['reg:linear'],
                 'nthread': [-1]
            }
        },
        {
            'class': RandomForestRegressor,
            'kwargs': {
                 'max_depth': [8, 16, None], 
                 'min_samples_split': [2, 4],
                 'min_samples_leaf': [1, 3],
                 'n_jobs': [-1]
            }
        }
    ]
}




class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, test, metadata): 
    fm = FeatureMaker(metadata)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)
    model = _MODELS[metadata['problem_type']]

    return x_train, y_train, x_test, y_test, model


def _weighted_f1(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    classes = list(report.keys())[:-3]
    proportion = [  report[i]['support'] / len(y_test) for i in classes]
    weighted_f1 = np.sum(list(map(lambda i, prop: report[i]['f1-score']* (1-prop)/(len(classes)-1), classes, proportion)))
    return weighted_f1 

@ignore_warnings(category=ConvergenceWarning)
def _evaluate_multi_classification(train, test, metadata):
    x_trains, y_trains, x_tests, y_tests = [], [], [], []
    for tr, te in zip(train, test):
        x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(tr, te, metadata)
        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)


    performance = []
    performance_std = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        unique_labels = np.unique(y_train)

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        results_std = []
        for param in tqdm(param_set):
            scores = []
            for x_train, y_train, x_test, y_test in zip(x_trains, y_trains, x_tests, y_tests):
                model = model_class(**param)

                try:
                    model.fit(x_train, y_train)
                except:
                    pass 
                
                if len(unique_labels) != len(np.unique(y_test)):
                    pred = [unique_labels[0]] * len(x_test)
                    pred_prob = np.array([1.] * len(x_test))
                else:
                    pred = model.predict(x_test)
                    pred_prob = model.predict_proba(x_test)

                macro_f1 = f1_score(y_test, pred, average='macro')
                weighted_f1 = _weighted_f1(y_test, pred)
                acc = accuracy_score(y_test, pred)

                # 3. auroc
                size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                rest_label = set(range(size)) - set(unique_labels)
                tmp = []
                j = 0
                for i in range(size):
                    if i in rest_label:
                        tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
                    else:
                        try:
                            tmp.append(pred_prob[:,[j]])
                        except:
                            tmp.append(pred_prob[:, np.newaxis])
                        j += 1
                roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp), multi_class='ovr')
                    
                scores.append(
                    {   
                        "name": model_repr,
                        "param": param,
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1,
                        "roc_auc": roc_auc, 
                        "accuracy": acc
                    }
                )
            scores_std = pd.DataFrame(scores).std(axis=0)
            scores = pd.DataFrame(scores).mean(axis=0)
            scores['name'] = model_repr
            scores_std['name'] = model_repr
            results.append(scores)
            results_std.append(scores_std)

        results = pd.DataFrame(results)
        results_std = pd.DataFrame(results_std)
        print(results)
        performance.append(
            {
                "name": results.name.max(),
                "weighted_f1": results.weighted_f1.max(),
                "roc_auc": results.roc_auc.max(),
                "accuracy": results.accuracy.max(),
                "macro_f1": results.macro_f1.max()
            }
        )

        performance_std.append(
            {
                "name": results.name.max(),
                "weighted_f1": results_std.weighted_f1[results.weighted_f1.idxmax()],
                "roc_auc": results_std.roc_auc[results.roc_auc.idxmax()],
                "accuracy": results_std.accuracy[results.accuracy.idxmax()],
                "macro_f1": results_std.macro_f1[results.macro_f1.idxmax()]
            }
        )
    return pd.DataFrame(performance), pd.DataFrame(performance_std)

@ignore_warnings(category=ConvergenceWarning)
def _evaluate_binary_classification(train, test, metadata):
    x_trains, y_trains, x_tests, y_tests = [], [], [], []
    for tr, te in zip(train, test):
        x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(tr, te, metadata)
        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)


    performance = []
    performance_std = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        unique_labels = np.unique(y_train)

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        results_std = []
        for param in tqdm(param_set):
            scores = []
            for x_train, y_train, x_test, y_test in zip(x_trains, y_trains, x_tests, y_tests):

                model = model_class(**param)
                
                try:
                    model.fit(x_train, y_train)
                except ValueError:
                    pass

                if len(unique_labels) == 1:
                    pred = [unique_labels[0]] * len(x_test)
                    pred_prob = np.array([1.] * len(x_test))
                else:
                    pred = model.predict(x_test)
                    pred_prob = model.predict_proba(x_test)

                binary_f1 = f1_score(y_test, pred, average='binary')
                weighted_f1 = _weighted_f1(y_test, pred)
                acc = accuracy_score(y_test, pred)
                precision = precision_score(y_test, pred, average='binary')
                recall = recall_score(y_test, pred, average='binary')
                macro_f1 = f1_score(y_test, pred, average='macro')

                # auroc
                size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                rest_label = set(range(size)) - set(unique_labels)
                tmp = []
                j = 0
                for i in range(size):
                    if i in rest_label:
                        tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
                    else:
                        try:
                            tmp.append(pred_prob[:,[j]])
                        except:
                            tmp.append(pred_prob[:, np.newaxis])
                        j += 1
                roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))

                scores.append(
                    {   
                        "name": model_repr,
                        "param": param,
                        "binary_f1": binary_f1,
                        "weighted_f1": weighted_f1,
                        "roc_auc": roc_auc, 
                        "accuracy": acc, 
                        "precision": precision, 
                        "recall": recall, 
                        "macro_f1": macro_f1
                    }
                )
            scores_std = pd.DataFrame(scores).std(axis=0)
            scores = pd.DataFrame(scores).mean(axis=0)
            scores['name'] = model_repr
            scores_std['name'] = model_repr

            results.append(scores)
            results_std.append(scores_std)

        results = pd.DataFrame(results)
        results_std = pd.DataFrame(results_std)
        print(results)
        performance.append(
            {
                "name": results.name.max(),
                "binary_f1": results.binary_f1.max(),
                "weighted_f1": results.weighted_f1.max(),
                "roc_auc": results.roc_auc.max(),
                "accuracy": results.accuracy.max(),
                "precision": results.precision.max(),
                "recall": results.recall.max(),
                "macro_f1": results.macro_f1.max()
            }
        )
        performance_std.append(
            {
                "name": results.name.max(),
                "binary_f1": results_std.binary_f1[results.binary_f1.idxmax()],
                "weighted_f1": results_std.weighted_f1[results.weighted_f1.idxmax()],
                "roc_auc": results_std.roc_auc[results.roc_auc.idxmax()],
                "accuracy": results_std.accuracy[results.accuracy.idxmax()],
                "precision": results_std.precision[results.precision.idxmax()],
                "recall": results_std.recall[results.recall.idxmax()],
                "macro_f1": results_std.macro_f1[results.macro_f1.idxmax()]
            }
        )
    return pd.DataFrame(performance), pd.DataFrame(performance_std)


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_regression(train, test, metadata):

    x_trains, y_trains, x_tests, y_tests = [], [], [], []
    for tr, te in zip(train, test):
        x_train, y_train, x_test, y_test, regressors = _prepare_ml_problem(tr, te, metadata)
        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)

    performance = []
    performance_std = []
    y_train = np.log(np.clip(y_train, 1, 20000))
    y_test = np.log(np.clip(y_test, 1, 20000))
    
    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        param_set = list(ParameterGrid(model_kwargs))
        results = []
        results_std = []

        for param in tqdm(param_set):
            scores = []
            for x_train, y_train, x_test, y_test in zip(x_trains, y_trains, x_tests, y_tests):

                model = model_class(**param)
                model.fit(x_train, y_train)
                pred = model.predict(x_test)

                r2 = r2_score(y_test, pred)
                explained_variance = explained_variance_score(y_test, pred)
                mean_squared = mean_squared_error(y_test, pred)
                root_mean_squared = mean_squared_error(y_test, pred, squared=False)

                mean_absolute = mean_absolute_error(y_test, pred)

                scores.append(
                    {   
                        "name": model_repr,
                        "param": param,
                        "r2": r2,
                        "explained_variance": explained_variance,
                        "mean_squared": mean_squared, 
                        "mean_absolute": mean_absolute, 
                        "rmse": root_mean_squared
                    }
                )
            scores_std = pd.DataFrame(scores).std(axis=0)
            scores = pd.DataFrame(scores).mean(axis=0)
            scores['name'] = model_repr
            scores_std['name'] = model_repr

            results.append(scores)
            results_std.append(scores_std)

        results = pd.DataFrame(results)
        results_std = pd.DataFrame(results_std)
        print(results)
        performance.append(
            {
                "name": results.name.max(),
                "r2": results.r2.max(),
                "explained_variance" : results.explained_variance.max(),
                "mean_squared_error" : results.mean_squared.min(),
                "mean_absolute_error" : results.mean_absolute.min(),
                "rmse": results.rmse.min()

            }
        )
        performance_std.append(
            {
                "name": results.name.max(),
                "r2": results_std.r2[results.r2.idxmax()],
                "explained_variance" : results_std.explained_variance[results.explained_variance.idxmax()],
                "mean_squared_error" : results_std.mean_squared[results.mean_squared.idxmin()],
                "mean_absolute_error" : results_std.mean_absolute[results.mean_absolute.idxmin()],
                "rmse" : results_std.rmse[results.rmse.idxmin()],
                

            }
        )

    return pd.DataFrame(performance), pd.DataFrame(performance_std)



_EVALUATORS = {
    'binary_classification': _evaluate_binary_classification,
    'multiclass_classification': _evaluate_multi_classification,
    'regression': _evaluate_regression
}



def compute_scores(test, synthesized_data, metadata):
    score, std = _EVALUATORS[metadata['problem_type']](synthesized_data, test, metadata) 
    return score, std
