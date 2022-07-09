from ast import Raise
from ctypes import Union
from genericpath import exists
from typing import List, Tuple

import pandas as pd
from pyparsing import Optional
from sklearn.preprocessing import LabelEncoder

from .internals import Setup
from .internals import _get_prep_pipeline


def prepare_data(
    train_data: pd.DataFrame,
    config: dict,
) -> Tuple:

    # Checking input
    _check_clf_target(train_data, config["modelling"]["target_clf"])
    _check_numeric_cols(train_data, config["modelling"]["numeric_cols"])
    _check_categorical_cols(train_data, config["modelling"]["categorical_cols"])

    # Get features and targets from config
    original_features = [
        *config["modelling"]["numeric_cols"],
        *config["modelling"]["categorical_cols"],
    ]
    target_clf = config["modelling"]["target_clf"]

    ## Preparing
    # Assembling preprocessing pipeline
    prep_pipe = _get_prep_pipeline()
    # Composing X_train
    X_train = prep_pipe.fit_transform(train_data[original_features])

    # Encoding target_clf
    y_train, y_clf_encoder = _encode_y_train(train_data[target_clf])

    return (
        Setup(
            X_train=X_train,
            y_clf_train=y_train,
            y_clf_encoder=y_clf_encoder,
            prep_pipe=prep_pipe,
        ),
        X_train.head(),
        y_train.head(),
    )


def _check_clf_target(train_data: pd.DataFrame, clf_col: str = None) -> None:
    """Raises LookupError if clf target column not in
    train_data dataframe.

    Parameters
    ----------
    train_data : pd.DataFrame
        Lookuperror

    clf_col : str
    """
    if clf_col:
        if clf_col not in train_data:
            raise LookupError(
                f"{clf_col}, which was provided as 'target_clf', is not in\
                    train_data dataframe. Check existence and spelling."
            )


def _check_numeric_cols(
    train_data: pd.DataFrame, numeric_cols: List[str] = None
) -> None:
    """Raises LookupError if one or more of the numeric
    columns listed in config are missing.

    Parameters
    ----------
    train_data : pd.DataFrame

    numeric_cols : List[str]
    """
    if numeric_cols:
        for col in numeric_cols:
            if col not in train_data:
                raise LookupError(
                    f"{col}, which was provided in 'numeric_cols', is not in\
                        train_data dataframe. Check existence and spelling."
                )


def _check_categorical_cols(
    train_data: pd.DataFrame, categorical_cols: List[str]
) -> None:
    """Raises LookupError if one or more of the categorical
    columns listed in config are missing.

    Parameters
    ----------
    train_data : pd.DataFrame

    categorical_cols : List[str]
    """
    if categorical_cols:
        for col in categorical_cols:
            if col not in train_data:
                raise LookupError(
                    f"{col}, which was provided in 'categorical_cols', is not in\
                        train_data dataframe. Check existence and spelling."
                )


def _encode_y_train(y: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    """Encodes target column of classification problems if required.
    Regardless of encoding, downcasts target column to save memory.

    Parameters
    ----------
    y : pd.Series
        Column to encode

    Returns
    -------
    Tuple[pd.Series, LabelEncoder]
        Encoded column and fitted LabelEncoder containing class labels.
    """
    if not _check_encoding_necessity(y):
        return pd.to_numeric(y, downcast="integer"), LabelEncoder()
    else:
        le = LabelEncoder().fit(y)
        y_trans = pd.Series(le.transform(y), name=y.name)
        # logger.info(f"Encoded target variable using classes: {*[(i, class_) for i, class_ in enumerate(le.classes_)]}")
        return pd.to_numeric(y_trans, downcast="integer"), le


def _check_encoding_necessity(y: pd.Series) -> bool:
    """Checks column for dtype integer.

    Parameters
    ----------
    y : pd.Series
        Target column of classification problem.

    Returns
    -------
    bool
        Boolean flag used for encoding
    """
    return True if (y.dtype == "O") else False
