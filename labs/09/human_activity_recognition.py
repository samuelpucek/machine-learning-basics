#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")
parser.add_argument("--grid_search", default=False, help="Grid Search Mode ")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        if args.grid_search:
            train_data, test_data, train_target, test_target = train_test_split(train.data, train.target, test_size=0.5,
                                                                                random_state=args.seed)

            # Grid Search

            # Create the parameter grid based on the results of random search
            param_grid = {
                'max_depth': [110, 115, 120],
                # 'max_features': ['log2', 'auto'],
                # 'min_samples_leaf': [1, 2, 3],
                # 'min_samples_split': [100],
                'n_estimators': [330, 335, 340, 345]  # [100, 200, 300, 1000]
            }
            # Create a based model
            rfc = RandomForestClassifier(n_jobs=-1)
            # Instantiate the grid search model
            grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid,
                                       cv=5, n_jobs=-1, verbose=2)

            # Fit the grid search to the data
            grid_search.fit(train_data, train_target)
            print('  ***')
            print('Results')
            print('  ***')
            print(grid_search.best_estimator_)
            print(grid_search.best_params_)
            print(grid_search.best_score_)

            # Model Accuracy
            training_accuracy = grid_search.score(train_data, train_target)
            testing_accuracy = grid_search.score(test_data, test_target)

            print(f'Training accuracy:\t{training_accuracy}')
            print(f'Testing accuracy:\t{testing_accuracy}')

            model = grid_search

        else:
            model = RandomForestClassifier(n_jobs=-1, max_depth=120, n_estimators=345)
            model.fit(train.data, train.target)
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate `predictions` with the test set predictions, either as a Python list of a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
