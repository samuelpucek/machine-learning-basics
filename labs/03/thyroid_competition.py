#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.pipeline
import sklearn.compose

import ssl

# SSL: CERTIFICATE_VERIFY_FAILED
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Pandas DataFrame
pd.set_option('display.max_columns', None)


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """

    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.5, type=float, help="Test size")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(train.data,
                                                                                                    train.target,
                                                                                                    test_size=args.test_size,
                                                                                                    random_state=args.seed)

        # Data preprocessing
        model = sklearn.pipeline.Pipeline([
            ('preprocessing', sklearn.compose.ColumnTransformer([
                ('onehot', sklearn.preprocessing.OneHotEncoder(sparse='False', handle_unknown='ignore'), slice(0, 15)),
                ('scaler', sklearn.preprocessing.MinMaxScaler(), slice(15, 21))])),
            ('poly', sklearn.preprocessing.PolynomialFeatures()),
            ('model', sklearn.linear_model.LogisticRegression(random_state=args.seed))
        ])

        params = {'poly__degree': [2],
                  'model__solver': ('newton-cg', 'lbfgs', 'liblinear'),
                  'model__C': np.geomspace(1, 100, num=10)
                  }

        grid = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=params, n_jobs=-1)
        grid.fit(train_data, train_target)

        # Find best model
        df = pd.DataFrame(data=grid.cv_results_)
        df.sort_values('rank_test_score', inplace=True)
        print(df[['rank_test_score', 'mean_test_score', 'param_model__C', 'param_model__solver', 'param_poly__degree']].head(10))
        print(grid.best_estimator_)
        print()
        print(f'Training score of the best model: {grid.best_score_}')
        print(grid.best_params_)
        print()
        print(f'Score of the winning model: {grid.score(test_data, test_target)}')

        # Best combination of params:
        # poly__degree = 2
        # model__solver = newton-cg
        # model__C = 200

        # Train a model on the given dataset and store it in `model`.
        model = sklearn.pipeline.Pipeline([
            ('preprocessing', sklearn.compose.ColumnTransformer([
                ('onehot', sklearn.preprocessing.OneHotEncoder(sparse='False', handle_unknown='ignore'), slice(0, 15)),
                ('scaler', sklearn.preprocessing.MinMaxScaler(), slice(15, 21))])),
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2)),
            ('model', sklearn.linear_model.LogisticRegression(random_state=args.seed, solver='newton-cg', C=200))
        ])

        # Train model on the full train dataset
        model.fit(train.data, train.target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
