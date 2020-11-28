#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
from sklearn import model_selection, preprocessing, linear_model, neural_network, pipeline, compose, metrics, ensemble

# SSL: CERTIFICATE_VERIFY_FAILED
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


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
parser.add_argument("--test_size", default=0.3, type=float, help="Test size")
parser.add_argument("--degree", default=2, type=int, help="Degree of polynomial features")
parser.add_argument("--model", default='gbt', type=str, help="Model")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        # Split data for training
        train_data, test_data, train_target, test_target = model_selection.train_test_split(train.data, train.target,
                                                                                            random_state=args.seed,
                                                                                            test_size=args.test_size)

        # Preprocessing
        ct = compose.ColumnTransformer([
            ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False), slice(0, 15)),
            ('scaler', preprocessing.MinMaxScaler(), slice(15, 21))])

        poly = preprocessing.PolynomialFeatures(degree=args.degree)

        if args.model == 'lr':
            model = pipeline.Pipeline([
                ('preprocessing', ct),
                ('poly', poly),
                ('model', linear_model.LogisticRegressionCV(Cs=np.geomspace(0.001, 1000, 7)))
            ])
        elif args.model == 'mlp':
            model = pipeline.Pipeline([
                ('preprocessing', ct),
                ('model', neural_network.MLPClassifier(hidden_layer_sizes=(100,),
                                                       learning_rate_init=0.01,
                                                       max_iter=500,
                                                       activation='relu'))
            ])
        elif args.model == 'gbt':
            model = ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=200).fit(train.data, train.target)

        # Fit and predict
        model.fit(train_data, train_target)
        predict_target = model.predict(test_data)

        # Compute error
        accuracy = metrics.accuracy_score(test_target, predict_target)
        f1score = metrics.f1_score(test_target, predict_target)

        # Print results
        print(f'Model: {args.model}')
        print('   ***')
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'F1 score: {f1score:.2f}')

        # Train model using whole dataset
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
