from typing import List, Tuple

import pandas as pd

from .internals import Setup


def prepare_data(
    train_data: pd.DataFrame,
    config: dict,
) -> Tuple:

    # Checking input
    _check_clf_target(train_data, config["modelling"]["target_clf"])
    _check_numeric_cols(train_data, config["modelling"]["numeric_cols"])

    ## Preparing
    # Composing X_train
    X_train = train_data[[*config["modelling"]["numeric_cols"]]]

    y_train = train_data[config["modelling"]["target_clf"]]

    return (
        Setup(
            X_train=X_train,
            y_clf_train=y_train,
        ),
        X_train.head(),
        y_train.head(),
    )


def _check_clf_target(train_data: pd.DataFrame, clf_col: str) -> None:
    """Raises LookupError if clf target column not in
    train_data dataframe.

    Parameters
    ----------
    train_data : pd.DataFrame
        Lookuperror
    clf_col : str
    """
    if clf_col not in train_data:
        raise LookupError(
            f"{clf_col}, which was provided as 'target_clf', is not in\
                 train_data dataframe. Check existence and spelling."
        )


def _check_numeric_cols(train_data: pd.DataFrame, numeric_cols: List[str]) -> None:
    """Raises LookupError if one or more of the numeric
    columns listed in config are missing.

    Parameters
    ----------
    train_data : pd.DataFrame

    numeric_cols : List[str]
    """
    for col in numeric_cols:
        if col not in train_data:
            raise LookupError(
                f"{col}, which was provided in 'numeric_cols', is not in\
                     train_data dataframe. Check existence and spelling."
            )
