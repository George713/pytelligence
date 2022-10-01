from typing import List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
)
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from . import _internals


def train_model(
    algorithm: str,
    setup: _internals.Setup,
    return_models: bool = False,
    feature_list: Optional[List[str]] = None,
    hyperparams: Optional[dict] = None,
) -> Tuple[object, pd.DataFrame]:
    """
    Trains a model instance using specified algorithm and
    calculates various metrics for it.

    Parameters
    ----------
    algorithm : str
        specifies which algorithm to use for training

    setup : Setup
        Dataclass containing the prepared data and further
        configurations.

    return_models: bool = False
        Flag for returning model instances trained on the
        entire training set.

    feature_list: Optional[List[str]] = None
        If provided, will train the model with provided
        feature list. Used within 'reduce_feature_space'
        functionality.

    hyperparams : Optional[dict]
        Hyperparameters to use with the given algorithm.
        Default `None` will use standard hyperparameters.

    Returns
    -------
    model : trained model instance

    metrics : pd.DataFrame
        contains one row listing the different metrics the
        model achieved
    """
    # Instantiate model instance
    model = _internals.get_model_instance(
        algorithm=algorithm,
        trial=None,
        hyperparams=hyperparams,
    )

    # Cross-validate model
    X_train = setup.X_train[feature_list] if feature_list else setup.X_train
    y_train = setup.y_clf_train

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        scoring={
            "accuracy": "accuracy",
            "precision": make_scorer(
                lambda *args, **kwargs: precision_score(
                    *args, **kwargs, zero_division=0
                )
            ),
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=-1,
    )

    # Fit model on full training dataset
    model = model.fit(X_train, y_train) if return_models else None

    # Aggregate metrics
    metrics = _internals.aggregate_metrics(cv_results, algorithm)

    return model, metrics
