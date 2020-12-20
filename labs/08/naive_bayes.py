#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Classifier
    if args.naive_bayes_type == 'gaussian':
        clf = sklearn.naive_bayes.GaussianNB()
    elif args.naive_bayes_type == 'multinomial':
        clf = sklearn.naive_bayes.MultinomialNB(alpha=args.alpha)
    elif args.naive_bayes_type == 'bernoulli':
        clf = sklearn.naive_bayes.BernoulliNB(alpha=args.alpha)
    else:
        print(f'naive_bayes_type={args.naive_bayes_type} is not supported')
        return

    # Fit
    clf.fit(train_data, train_target)
    # Predict
    pred_target = clf.predict(test_data)
    # Accuracy
    test_accuracy = sklearn.metrics.accuracy_score(test_target, pred_target)

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
