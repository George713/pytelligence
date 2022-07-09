from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

    y_clf_encoder: LabelEncoder
        LabelEncoder used for encoding classification target.
    """

    X_train: pd.DataFrame
    y_clf_train: pd.Series
    y_clf_encoder: LabelEncoder
