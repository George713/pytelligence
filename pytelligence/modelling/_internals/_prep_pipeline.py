import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(f"stream.{__name__}")


def get_prep_pipeline(config: dict) -> Pipeline:
    """Return unfitted preprocessing pipeline.

    Preprocessing steps include:
      1) Standardization of numeric columns
      2) One-Hot-Encoding of 'object' columns

    Parameters
    ----------
    config : Dict
        Configuration read from provided config.yml

    Returns
    -------
    Pipeline
        Unfitted preprocessing pipeline.
    """
    steps = [
        # StandardScaler (optional)
        (
            "scaler",
            CustomStandardScaler(numeric_cols=config["modelling"]["numeric_cols"]),
        )
        if config["modelling"]["feature_scaling"]
        else "skip",
        # OneHotEncoder
        ("ohe", OHE()),
    ]

    prep_pipe = Pipeline(steps=[step for step in steps if step != "skip"])
    step_names = [step[0] for step in prep_pipe.steps]
    logger.info(f"Created preprocessing pipeline with following steps: {step_names}")
    return prep_pipe


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """Transformer class applying scaling to numeric features."""

    def __init__(self, numeric_cols):
        self.scaler = StandardScaler()
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        self.scaler.fit(X.loc[:, self.numeric_cols])
        return self

    def transform(self, X, y=None):
        X.loc[:, self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X


class OHE(BaseEstimator, TransformerMixin):
    """Transformer class performing One-Hot-Encoding of 'object' typed columns."""

    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.col_names = None

    def fit(self, X, y=None):
        categorical_data = X.select_dtypes(include=("object"))
        if len(categorical_data.columns) > 0:
            self.ohe.fit(categorical_data)
            self.col_names = self.ohe.get_feature_names_out(categorical_data.columns)
        return self

    def transform(self, X, y=None):
        # Checking whether encoder was fitted
        if self.col_names is not None:
            non_categorical_data = X.select_dtypes(exclude=("object"))
            categorical_data = X.select_dtypes(include=("object"))
            ohe_data = self.ohe.transform(categorical_data).toarray()
            ohe_df = pd.DataFrame(
                data=ohe_data, columns=self.col_names, index=non_categorical_data.index
            ).astype("uint8")
            return pd.concat([non_categorical_data, ohe_df], axis=1)
        else:
            return X
