import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from pycarrot.modelling.internals import Setup

setup = Setup(
    X_train=pd.DataFrame(),
    y_clf_train=pd.Series(),
    y_clf_encoder=LabelEncoder(),
    prep_pipe=Pipeline(steps=[]),
)


def test_class_attributes():
    assert all(
        hasattr(setup, attribute_name)
        for attribute_name in [
            "X_train",
            "y_clf_train",
            "y_clf_encoder",
            "prep_pipe",
        ]
    )
