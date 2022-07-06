from dataclasses import dataclass
import pandas as pd


@dataclass
class Setup:
    """
    Container class to be returned when using
    modelling.prepare_data(). It holds all the information
    acquired during this preparation step.

    Attributes
    ----------
    X_train: pd.DataFrame
        Training dataset without target.

    y_clf_train: pd.Series
        Classifier target of training dataset.
    """

    X_train: pd.DataFrame
    y_clf_train: pd.Series
