

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score, classification_report, f1_score, make_scorer
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# demo utils
from utils import load_ptsd_data


def run_model():

    # get the texts and their corresponding labels
    texts, labels = load_ptsd_data()

    estimators = [
        ('count_vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())]
    pipe = Pipeline(estimators)

    param_grid = [
        {'count_vect__max_features': np.arange(4000, 6000, 500),
         'count_vect__min_df': np.arange(0.0, 0.1, 0.05),
         'count_vect__max_df': np.arange(0.6, 0.8, 0.05),
         'tfidf': [None, TfidfTransformer()]}
    ]

    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=make_scorer(f1_score, average='weighted'),
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(texts, labels)
    prediction = grid_search.predict(texts)

    print(grid_search.best_params_)

    print("results")
    print(f1_score(labels, prediction, average='weighted'))
    print(classification_report(labels, prediction))
    print(confusion_matrix(labels, prediction))
    print(recall_score(labels, prediction))
    print(accuracy_score(labels, prediction))


if __name__ == '__main__':

    run_model()
