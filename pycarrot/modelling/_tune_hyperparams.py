"""
Contains logic for hyperparameter tuning.

Example
-------
>>> import pycarrot as pc
>>> from sklearn.datasets import load_breast_cancer

>>> bc = load_breast_cancer()
>>> X = pd.DataFrame(bc.data, columns=bc.feature_names)
>>> y = pd.Series(bc.target, name="class")
>>> df = pd.concat([X, y], axis=1)

>>> config = pc.init_config("./config_bc.yml")

>>> setup, X_sample, y_sample = pc.modelling.prepare_data(
        train_data=df,
        config=config,
    )

>>> compare_df, algo_list, model_list = (
            pc.modelling.tune_hyperparams(
                   setup=setup,
                   include=["lr", "knn"],
                   optimize="f1",
                   n_trials=50,
                   return_models=True,
            )
    )
"""
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import cross_val_score
import optuna

from . import _internals


def tune_hyperparams(
    setup: _internals.Setup,
    include: List[str],
    optimize: str,
    n_trials: int = 20,
    return_models: bool = False,
) -> Tuple[pd.DataFrame, list, list]:
    """Tunes the algorithms provided in the `include` parameter.

    Example
    -------
    >>> compare_df, algo_list, model_list = (
                pc.modelling.tune_hyperparams(
                    setup=setup,
                    include=["lr", "knn"],
                    optimize="f1",
                    n_trials=50,
                    return_models=True,
                )
        )

    Parameters
    ----------
    setup : Setup
        Dataclass containing the prepared data and further
        configurations.

    include : List[str]
        List of identifier strings for algorithms to tune.

    optimize : str
        String identifier of metric to optimize for.

    n_trials : int, optional
        Number of hyperparameter combinations to evaluate,
        by default 20

    return_models : bool, optional
        Flag for training models on the entire dateset using the
        best hyperparameter combinations found, by default False

    Returns
    -------
    Tuple[pd.DataFrame, list, list]
        compare_df : pd.DataFrame
            sorted overview of algorithm performance

        algo_list : list
            List of algorithms ordered by sort metric.

        model_list : list
            Trained model instance if return_models == True.
            Otherwise returns list of None.
    """
    # Checking inputs
    _internals.check_include(algo_list=include)
    _internals.check_metric(metric=optimize)

    # # Preparing empty compare_df and model_dict
    # # with populating occuring later
    # compare_df = _prepare_compare_df()
    # model_dict = {}

    for algorithm in include:
        objective = _get_objective_function(algorithm=algorithm, optimize=optimize)
        # study = _create_study
        # _study.optimize(objective, n_trials=n_trials)
        # metric, hyperparams = _get_best_result(study)
        # compare_df.loc[len(compare_df)] = [
        #     algorithm,
        #     metric,
        #     hyperparams,
        # ]

    return pd.DataFrame(), [], []


def _get_objective_function(algorithm: str, optimize: str) -> object:
    """Returns model specific objective function for usage in optuna's
    stduy.optimize() function.

    Parameters
    ----------
    algorithm : str
        Model algorithm to use in objective function.

    optimize : str
        Metric to optimize for.

    Returns
    -------
    object
        `objective function` object
    """

    def objective_fn(trial: optuna.Trial):
        model = _internals.get_model_instance(algorithm=algorithm, trial=trial)
        # cv_scores = cross_val_score(model, setup.X_train, setup.y_clf_train, scoring=optimize, n_jobs=-1, error_score='raise')

        # return cv_scores.mean()
        pass

    return objective_fn
