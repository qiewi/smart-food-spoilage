"""4 model klasik + grid hyperparameter bersitasi + fungsi tuning (inner StratifiedKFold)."""

import warnings

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .data import make_preprocessor

# liblinear + penalty menghasilkan FutureWarning kosmetik di sklearn 1.8; senyapkan.
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*")


def model_specs():
    """(nama, estimator, param_grid, search) untuk tiap model. Grid key prefiks 'clf__'."""
    rs = config.RANDOM_STATE
    return [
        ("Logistic Regression", LogisticRegression(solver="liblinear"), {
            "clf__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l1", "l2"],
            "clf__class_weight": [None, "balanced"],
        }, "grid"),
        ("Decision Tree", DecisionTreeClassifier(random_state=rs), {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 3, 5, 7, 9, 15],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__class_weight": [None, "balanced"],
        }, "grid"),
        ("KNN", KNeighborsClassifier(), {
            "clf__n_neighbors": [1, 3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        }, "grid"),
        ("Random Forest", RandomForestClassifier(random_state=rs, n_jobs=-1), {
            "clf__n_estimators": [50, 100, 200, 500, 1000],
            "clf__max_depth": [None, 5, 7, 10],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", "log2"],
            "clf__criterion": ["gini", "entropy"],
            "clf__class_weight": [None, "balanced"],
        }, "random"),
    ]


def tune(est, grid, kind, X, y):
    """Tune 1 model via inner StratifiedKFold(3), refit F1. Return (best_estimator, best_params)."""
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clone(est))])
    try:
        pipe.set_params(clf__n_jobs=1)
    except ValueError:
        pass
    cv = StratifiedKFold(3, shuffle=True, random_state=config.RANDOM_STATE)
    if kind == "random":
        search = RandomizedSearchCV(pipe, grid, n_iter=config.RF_N_ITER, scoring=config.SCORING,
                                    refit=config.REFIT_METRIC, cv=cv,
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    else:
        search = GridSearchCV(pipe, grid, scoring=config.SCORING, refit=config.REFIT_METRIC,
                              cv=cv, n_jobs=-1)
    search.fit(X, y)
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params
