#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data,
                                                                                                dataset.target,
                                                                                                test_size=args.test_size,
                                                                                                random_state=args.seed)

    # Pipeline
    pipe = sklearn.pipeline.Pipeline(
        [('scaler',
          sklearn.preprocessing.MinMaxScaler(feature_range=(np.min(dataset.data), np.max(dataset.data)))),
         ('poly', sklearn.preprocessing.PolynomialFeatures()),
         ('model', sklearn.linear_model.LogisticRegression(random_state=args.seed))
         ])

    # Params
    parameters = {'poly__degree': [1, 2], 'model__solver': ('sag', 'lbfgs'), 'model__C': [0.01, 1, 100]}

    # Grid Model
    model = sklearn.model_selection.GridSearchCV(pipe, parameters,
                                                 cv=sklearn.model_selection.StratifiedKFold(5), n_jobs=-1)

    # Fit the Model(s)
    model.fit(train_data, train_target)

    return model.score(test_data, test_target)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))
