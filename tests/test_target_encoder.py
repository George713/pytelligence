"""Tests for encoding functionality of target variable."""
import pandas as pd
import pytest

import pytelligence as pt


@pytest.fixture
def df_clf():
    return pd.DataFrame(
        data={
            "num_col": [i for i in range(100)],
            "cat_col": [str(i) for i in range(100)],
            "target_clf": ["bad"] * 50 + ["good"] * 50,
        }
    )


@pytest.fixture
def X_train(df_clf):
    return df_clf[["num_col", "cat_col"]]


@pytest.fixture
def setup(df_clf):
    return pt.modelling.prepare_data(
        train_data=df_clf,
        config_path="./tests/config_test.yml",
    )[0]


@pytest.fixture
def trained_model(setup):
    return pt.modelling.compare_algorithms(
        setup=setup,
        include=[
            "nb",
        ],
        sort="f1",
        return_models=True,
    )[2][0]


@pytest.fixture
def pipeline_w_encoder(setup, trained_model):
    return pt.modelling._export_model._combine_pipeline_and_model(
        prep_pipe=setup.prep_pipe,
        model=trained_model,
        y_clf_encoder=setup.y_clf_encoder,
    )


@pytest.fixture
def pipeline_wo_encoder(setup, trained_model):
    return pt.modelling._export_model._combine_pipeline_and_model(
        prep_pipe=setup.prep_pipe,
        model=trained_model,
        y_clf_encoder=None,
    )


def test_with_encoding(pipeline_w_encoder, X_train, df_clf):
    assert set(pipeline_w_encoder.predict(X_train).tolist()) == set(
        df_clf["target_clf"].unique()
    )


def test_without_encoding(pipeline_wo_encoder, X_train, df_clf):
    assert set(pipeline_wo_encoder.predict(X_train).tolist()) == set(
        range(0, df_clf["target_clf"].nunique())
    )
