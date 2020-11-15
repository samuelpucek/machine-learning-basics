#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="boston", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data,
                                                                                                dataset.target,
                                                                                                test_size=args.test_size,
                                                                                                random_state=args.seed)
    categorical_columns = np.all(train_data.astype(int) == train_data, axis=0)

    preprocess = sklearn.pipeline.Pipeline([
        ('preprocess', sklearn.compose.ColumnTransformer([
            ('onehot', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_columns),
            ('scaler', sklearn.preprocessing.StandardScaler(), ~categorical_columns)
        ])),
        ('poly', sklearn.preprocessing.PolynomialFeatures(2, include_bias=False))
    ])

    train_data = preprocess.fit_transform(train_data)
    test_data = preprocess.transform(test_data)

    return train_data, test_data


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 60))))
