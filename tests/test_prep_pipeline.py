import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import pytelligence as pt


@pytest.fixture
def test_df():
    return pd.DataFrame(
        data={
            "num_col": range(1, 10),
            "cat_col1": [str(i) for i in range(1, 10)],
            "cat_col2": [str(i) + str(i) for i in range(1, 10)],
        }
    )


@pytest.fixture
def config():
    return pt.modelling._prepare_data._init_config(path="./tests/config_test.yml")


@pytest.fixture
def prep_pipe(config):
    return pt.modelling._internals.get_prep_pipeline(config=config)


@pytest.fixture
def fitted_prep_pipe(config, test_df):
    return pt.modelling._internals.get_prep_pipeline(config=config).fit(test_df)


@pytest.fixture
def ohe(config):
    return pt.modelling._internals.get_prep_pipeline(config=config).steps[-1][1]


@pytest.fixture
def fitted_ohe(fitted_prep_pipe):
    return fitted_prep_pipe.steps[-1][1]


def testget_prep_pipeline(prep_pipe):
    """Checks return type."""
    assert type(prep_pipe) == Pipeline


def test_step_ohe(ohe):
    """Checks type of transformer in last pipeline step."""
    assert type(ohe) == pt.modelling._internals.OHE


def test_ohe_attributes(ohe):
    """Checks attributes of ohe transformer."""
    assert hasattr(ohe, "ohe")
    assert hasattr(ohe, "col_names")


def test_ohe_unfitted_state(ohe):
    """Checks ohe.col_names for value before fitting."""
    assert ohe.col_names is None


def test_ohe_fit(fitted_ohe):
    """Checks if ohe.fit is functional."""
    assert fitted_ohe.col_names is not None


def test_fitted_ohe_attr_content(fitted_ohe):
    """Checks if column names have been correctly encoded."""
    for i in range(1, 10):
        assert f"cat_col1_{i}" in fitted_ohe.col_names
        assert f"cat_col2_{i}{i}" in fitted_ohe.col_names


def test_prep_pipe_transform(fitted_prep_pipe, test_df):
    """Checks return type of fitted_prep_pipe.transform()."""
    result = fitted_prep_pipe.transform(test_df)
    assert type(result) == pd.DataFrame


def test_df_after_ohe_transform(fitted_prep_pipe, test_df):
    """Checks transformation via ohe transformer."""
    result = fitted_prep_pipe.transform(test_df)
    assert "num_col" in result.columns
    for i in range(1, 10):
        assert f"cat_col1_{i}" in result.columns
        assert f"cat_col2_{i}{i}" in result.columns


def test_dtype_of_ohe_transform(fitted_prep_pipe, test_df):
    """Checks for dtype uint8 of ohe transformation."""
    result = (
        fitted_prep_pipe.transform(test_df).select_dtypes(include=("uint8")).columns
    )
    assert all("cat_col" in col for col in result)
