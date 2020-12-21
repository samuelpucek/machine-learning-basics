#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrapping", default=False, action="store_true", help="Perform data bootstrapping")
parser.add_argument("--feature_subsampling", default=1, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    clf = RandomForestClassifier(n_estimators=args.trees,
                                 criterion='entropy',
                                 max_depth=args.max_depth,
                                 bootstrap=args.bootstrapping,
                                 max_features=args.feature_subsampling
                                 )

    clf.fit(train_data, train_target)

    # Measure the training and testing accuracy.
    train_accuracy = clf.score(train_data, train_target)
    test_accuracy = clf.score(test_data, test_target)

    return train_accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
