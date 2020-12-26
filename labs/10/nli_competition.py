#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")
parser.add_argument("--grid_search", default='yes', type=str)
parser.add_argument("--classifier", default='nb', type=str)


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset("nli_dataset.train.txt")
        dev = Dataset("nli_dataset.dev.txt")

        train_data, test_data, train_target, test_target = train_test_split(train.data, train.target,
                                                                            random_state=args.seed, test_size=0.3)

        # Classifier
        if args.classifier == 'sgd':  # .999 / .718
            clf = SGDClassifier()
        elif args.classifier == 'mlp':  # .999 / .732
            clf = MLPClassifier(hidden_layer_sizes=(300,))
        elif args.classifier == 'svm':  # .999 / .708
            clf = SVC()
        else:  # .947 / .676
            clf = MultinomialNB()

        # Pipeline
        pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', clf)
        ])

        params = {'vect__analyzer': ['word'],
                  'vect__lowercase': [False],
                  'vect__ngram_range': [(1, 2)],
                  'vect__max_features': [15000],
                  'vect__min_df': [15],
                  'vect__max_df': [0.03]
                  }

        model = GridSearchCV(estimator=pipeline, param_grid=params, n_jobs=-1, verbose=0)

        # Fit
        model.fit(train_data, train_target)
        # model.fit(train.data, train.target)

        # print(model.best_score_)
        # print(model.best_estimator_)
        # print(model.best_params_)

        # Accuracy
        print(f'Model Score - training data:\t{model.score(train_data, train_target)}')
        print(f'Model Score - testing data:\t{model.score(test_data, test_target)}')

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate `predictions` with the test set predictions
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
