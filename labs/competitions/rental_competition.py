#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np

from sklearn import model_selection, preprocessing, linear_model, pipeline, compose, metrics, neural_network

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
parser.add_argument("--model", default="mlp", type=str, help="Used model")
parser.add_argument("--degree", default=2, type=int, help="Degree of poly features")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Data
        train_data, test_data, train_target, test_target = model_selection.train_test_split(train.data, train.target,
                                                                                            random_state=args.seed,
                                                                                            test_size=0.4)
        # Preprocessing
        column_transformer = compose.ColumnTransformer(
            [('categories', preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False), slice(0, 8)),
             ('scaler', preprocessing.MinMaxScaler(), slice(9, 12))])
        poly = preprocessing.PolynomialFeatures(degree=args.degree)

        # Appropriate model
        if args.model == 'lr':
            model = linear_model.LinearRegression()
            pipe = pipeline.Pipeline([('preprocess', column_transformer), ('poly', poly), ('model', model)])
        elif args.model == 'ridgecv':
            model = linear_model.RidgeCV(alphas=np.arange(0.1, 10.1, 0.1))
            pipe = pipeline.Pipeline([('preprocess', column_transformer), ('poly', poly), ('model', model)])
        elif args.model == 'poisson':
            model = linear_model.PoissonRegressor(max_iter=900)
            pipe = pipeline.Pipeline([('preprocess', column_transformer), ('poly', poly), ('model', model)])
        elif args.model == 'mlp':
            model = neural_network.MLPRegressor(max_iter=1000)
            model = neural_network.MLPRegressor(learning_rate_init=0.01, max_iter=200,
                                                hidden_layer_sizes=(300, 200, 100),
                                                activation="relu", solver="adam")
            pipe = pipeline.Pipeline([('preprocess', column_transformer), ('model', model)])

        # Fitting and prediction
        pipe.fit(train_data, train_target)
        predict_target = pipe.predict(test_data)

        # Evaluation
        score = pipe.score(test_data, test_target)
        rmse = metrics.mean_squared_error(test_target, predict_target, squared=False)

        # Results
        print(f'Used model: {args.model}')
        print(f'Poly degrees: {args.degree}')
        print('   ***')
        print(f'Score: {score:.2f}%')
        print(f'Root mean square error: {rmse:.2f}')

        # Train the winning model on the given dataset and store it in `model`.
        model = pipe
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
