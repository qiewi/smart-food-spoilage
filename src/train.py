"""Build the four model pipelines and tune them with grouped CV.

Hyperparameter search spaces follow cited references (Hasan 2020; Sarno 2023;
arXiv:2310.14629) and scikit-learn conventions; the final values are selected
empirically via cross-validation (GridSearchCV / RandomizedSearchCV).
"""

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedGroupKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .preprocess import make_preprocessor

# LogisticRegression(penalty=...) with solver='liblinear' (per Hasan 2020) still
# works in scikit-learn 1.8 but emits a cosmetic deprecation FutureWarning
# (removed only in 1.10). Silence it; the l1/l2 selection is statistically valid.
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*",
                        category=FutureWarning)

# Scoring suite. Recall/precision/F1 default to the positive class (spoiled=1).
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# Refit on F1, not recall: recall alone rewards a trivial all-spoiled
# classifier. Final model selection still leads with recall (see evaluate.py).
REFIT_METRIC = "f1"
# RandomForest search space is large (1440 combos); sample it instead of a full
# grid to keep tuning tractable while still covering the cited ranges.
RF_N_ITER = 80


def _model_specs():
    """(name, estimator, param_grid, search) for each classifier.

    Grid keys are prefixed with `clf__` to target the classifier step.
    `search` is "grid" (exhaustive) or "random" (sampled). `class_weight`
    is explored ([None, 'balanced']) to test handling of class imbalance.
    """
    rs = config.RANDOM_STATE
    return [
        (
            "Logistic Regression",
            LogisticRegression(solver="liblinear"),
            {
                "clf__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "clf__penalty": ["l1", "l2"],
                "clf__class_weight": [None, "balanced"],
            },
            "grid",
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=rs),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 3, 5, 7, 9, 15],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 3, 5],
                "clf__class_weight": [None, "balanced"],
            },
            "grid",
        ),
        (
            "KNN",
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [1, 3, 5, 7, 9, 11],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
            "grid",
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {
                "clf__n_estimators": [50, 100, 200, 500, 1000],
                "clf__max_depth": [None, 5, 7, 10],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 3, 5],
                "clf__max_features": ["sqrt", "log2"],
                "clf__criterion": ["gini", "entropy"],
                "clf__class_weight": [None, "balanced"],
            },
            "random",
        ),
    ]


def make_cv() -> StratifiedGroupKFold:
    """Grouped, stratified CV: whole runs held out per fold (leakage-safe)."""
    return StratifiedGroupKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )


def tune_models(X, y, groups):
    """Tune each model with grouped CV. Returns {name: fitted search object}.

    RandomForest uses RandomizedSearchCV (large space); the rest use exhaustive
    GridSearchCV. Both expose best_estimator_/best_params_/cv_results_.
    """
    cv = make_cv()
    results = {}
    for name, estimator, grid, search_kind in _model_specs():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
        if search_kind == "random":
            search = RandomizedSearchCV(
                pipe,
                param_distributions=grid,
                n_iter=RF_N_ITER,
                scoring=SCORING,
                refit=REFIT_METRIC,
                cv=cv,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            search = GridSearchCV(
                pipe,
                param_grid=grid,
                scoring=SCORING,
                refit=REFIT_METRIC,
                cv=cv,
                n_jobs=-1,
            )
        search.fit(X, y, groups=groups)
        results[name] = search
    return results


def cv_score_summary(search) -> dict:
    """Mean +/- std across folds for the best params (PRD checklist item 7)."""
    cvres = search.cv_results_
    best = search.best_index_
    summary = {"best_params": search.best_params_}
    for metric in SCORING:
        summary[f"{metric}_mean"] = float(cvres[f"mean_test_{metric}"][best])
        summary[f"{metric}_std"] = float(cvres[f"std_test_{metric}"][best])
    return summary


def format_cv_summary(name, summary) -> str:
    lines = [f"{name}  (best params: {summary['best_params']})"]
    for metric in SCORING:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        lines.append(f"    {metric:10s}: {mean:.3f} +/- {std:.3f}")
    return "\n".join(lines)
