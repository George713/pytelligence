import pandas as pd

from pycarrot.modelling._setup import Setup

setup = Setup(
    X_train=pd.DataFrame(),
    y_clf_train=pd.Series(),
)


def test_class_attributes():
    assert all(
        hasattr(setup, attribute_name)
        for attribute_name in [
            "X_train",
            "y_clf_train",
        ]
    )
