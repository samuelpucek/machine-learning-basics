#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # The input data are in dataset.data, targets are in dataset.target.

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # Data
    bias = np.ones([dataset.data.shape[0], 1])
    regresors = np.concatenate((dataset.data, bias), axis=1)
    targets = dataset.target

    # Split dataset
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(regresors, targets,
                                                                                test_size=args.test_size,
                                                                                random_state=args.seed)

    # Weights computation
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    # Prediction for X_test
    y_pred = X_test @ w

    # Compute root mean square error on the test set predictions
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    return np.sqrt(mse)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
