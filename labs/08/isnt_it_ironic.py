#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.zip",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with zipfile.ZipFile(name, "r") as dataset_file:
            with dataset_file.open(os.path.basename(name).replace(".zip", ".txt"), "r") as train_file:
                for line in train_file:
                    label, text = line.decode("utf-8").rstrip("\n").split("\t")
                    self.data.append(text)
                    self.target.append(int(label))
        self.target = np.array(self.target, np.int32)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Split data
        train_data, test_data, train_target, test_target = train_test_split(train.data, train.target,
                                                                            test_size=args.test_size,
                                                                            random_state=args.seed)

        # Model
        pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', BernoulliNB()),
            # ('clf', MultinomialNB()),
        ])

        pipe_bernoulli = Pipeline([
            ('vect', TfidfVectorizer(analyzer='word', lowercase=True, max_features=2500)),
            ('clf', BernoulliNB(alpha=7.74))
        ])

        pipe_multinomial = Pipeline([
            ('vect', TfidfVectorizer(analyzer='word', lowercase=False, max_features=5000)),
            ('clf', MultinomialNB(alpha=7.74))
        ])

        params = {
            'vect__analyzer': ('word', 'char', 'char_wb'),
            'vect__lowercase': [True, False],
            'vect__max_features': np.geomspace(10, 5000, 10, dtype=int),
            'clf__alpha': np.geomspace(0.001, 100, 10),
        }

        model = GridSearchCV(pipeline, params, n_jobs=-1, verbose=10)
        # Fit model
        model.fit(train_data, train_target)

        print('Model Selection')
        print(model.best_score_)
        print(model.best_params_)
        print(model.best_estimator_)
        print()
        print('   ***')
        print()

        # Predict
        pred_target = model.predict(test_data)

        # Score
        model_score = model.score(test_data, test_target)
        f1 = f1_score(test_target, pred_target)
        accuracy = accuracy_score(test_target, pred_target)

        print(f'Model score:\t{model_score}')
        print(f'F1 score:\t{f1}')
        print(f'Accuracy score:\t{accuracy}')

        # Final training
        # model.fit(train.data, train.target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate `predictions`
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
