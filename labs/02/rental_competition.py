#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
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


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: sprint, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rentals in the given hour.
    """

    def __init__(self,
                 name="rental_competition.train.npz",
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
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.3, type=float, help="Size of test subset")
parser.add_argument("--learning_rate", default=0.0026, type=float, help="Size of test subset")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(train.data,
                                                                                                    train.target,
                                                                                                    test_size=args.test_size,
                                                                                                    random_state=args.seed)

        # SGD Model
        alphas = np.geomspace(0.0015, 0.0036, num=20)
        rmses = []
        best_learning_rate = 0
        best_rmse = 1000
        for alpha in alphas:
            model = sklearn.pipeline.Pipeline(
                [
                    ('preprocess', sklearn.compose.ColumnTransformer(
                        [('onehot', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore'),
                          slice(0, 8))])),
                    ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
                    # ('linear_regression', sklearn.linear_model.LinearRegression())
                    ('linear_SGD', sklearn.linear_model.SGDRegressor(loss='squared_loss',
                                                                     penalty='l2',
                                                                     # alpha=args.learning_rate,
                                                                     alpha=alpha,
                                                                     random_state=args.seed
                                                                     ))
                ])

            model.fit(train_data, train_target)
            pred_target = model.predict(test_data)

            rmse = sklearn.metrics.mean_squared_error(test_target, pred_target, squared=False)
            rmses.append(rmse)
            # Find the best learning rate
            if rmse < best_rmse:
                best_rmse = rmse
                best_learning_rate = alpha
            print(f'Learning rate: {alpha:.4f} \t RMSE: {rmse:,.2f}')

        print()
        print('Best Learning rate and best RMSE')
        print(f'Learning rate: {best_learning_rate:.4f} \t RMSE: {best_rmse:,.2f}')

        # Final model
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
