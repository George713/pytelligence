import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from pycarrot.modelling._internals import Setup

setup = Setup(
    X_train=pd.DataFrame(),
    y_clf_train=pd.Series(dtype="int"),
    y_clf_encoder=LabelEncoder(),
    feature_scaling=True,
    prep_pipe=Pipeline(steps=[]),
)


def test_class_attributes():
    assert all(
        hasattr(setup, attribute_name)
        for attribute_name in [
            "X_train",
            "y_clf_train",
            "y_clf_encoder",
            "feature_scaling",
            "prep_pipe",
        ]
    )
