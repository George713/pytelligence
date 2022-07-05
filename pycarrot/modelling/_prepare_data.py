from logging import raiseExceptions
from typing import Tuple

import pandas as pd

from ._setup import Setup


def prepare_data(
    train_data: pd.DataFrame,
    config: dict,
) -> Tuple:

    ## Checking input
    # Checking for classifier target column presence
    if config["modelling"]["target_clf"] not in train_data:
        raise LookupError(
            f"{config['modelling']['target_clf']} not in train_data dataframe. Check existence and spelling."
        )

    # Checking for numeric column presence
    for col in config["modelling"]["numeric_cols"]:
        if col not in train_data:
            raise LookupError(
                f"{col} not in train_data dataframe. Check existence and spelling."
            )

    ## Preparing
    # Composing X_train
    X_train = train_data[
        [*config["modelling"]["numeric_cols"]]
    ]

    y_train = train_data[config["modelling"]["target_clf"]]

    return (
        Setup(
            X_train=X_train,
            y_clf_train=y_train,
        ),
        X_train.head(),
        y_train.head(),
    )
