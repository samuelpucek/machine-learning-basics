#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

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
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """

    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)
        self.data = self.data.reshape([-1, 28 * 28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--models", default=1, type=int, help="Model to train")
parser.add_argument("--iterations", default=15, type=int, help="Training iterations")
parser.add_argument("--augment", default=False, action="store_true", help="Augment during training")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Data
        train_data, test_data, train_target, test_target = model_selection.train_test_split(train.data, train.target,
                                                                                            random_state=args.seed,
                                                                                            test_size=0.5)

        # MLP model
        model = pipeline.Pipeline([
            ('preprocessing', preprocessing.MinMaxScaler()),
            ('MLPs', ensemble.VotingClassifier([
                ('MLP{}'.format(i), neural_network.MLPClassifier(tol=0, verbose=True, alpha=0,
                                                                 hidden_layer_sizes=(500),
                                                                 max_iter=1 if args.augment else args.iterations))
                for i in range(args.models)
            ], voting='soft'))
        ])

        # Model accuracy
        # model.fit(test_data, test_target)
        # predict_target = model.predict(test_data)
        # accuracy = metrics.accuracy_score(test_target, predict_target)
        # print(f'Model accuracy: {accuracy:.3f}%')

        # Train a model on the given dataset and store it in `model`.
        model.fit(train.data, train.target)

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        # mlp._optimizer = None
        # for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
        for mlp in model["MLPs"].estimators_:
            mlp._optimizer = None
            for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

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
