import pandas as pd
from sklearn.pipeline import Pipeline

import pycarrot as pc

prep_pipe = pc.modelling._internals.get_prep_pipeline()
ohe = prep_pipe.steps[-1][1]

test_df = pd.DataFrame(
    data={
        "num_col": range(1, 10),
        "cat_col1": [str(i) for i in range(1, 10)],
        "cat_col2": [str(i) + str(i) for i in range(1, 10)],
    }
)


def testget_prep_pipeline():
    """Checks return type."""
    assert type(prep_pipe) == Pipeline


def test_step_ohe():
    """Checks type of transformer in last pipeline step."""
    assert type(ohe) == pc.modelling._internals.OHE


def test_ohe_attributes():
    """Checks attributes of ohe transformer."""
    assert hasattr(ohe, "ohe")
    assert hasattr(ohe, "col_names")


def test_ohe_unfitted_state():
    """Checks ohe.col_names for value before fitting."""
    assert ohe.col_names is None


# Fitting of pipeline occurs in this test
def test_ohe_fit():
    """Checks if ohe.fit is functional."""
    prep_pipe.fit(test_df)
    assert ohe.col_names is not None


def test_fitted_ohe_attr_content():
    """Checks if column names have been correctly encoded."""
    for i in range(1, 10):
        assert f"cat_col1_{i}" in ohe.col_names
        assert f"cat_col2_{i}{i}" in ohe.col_names


def test_prep_pipe_transform():
    """Checks return type of prep_pipe.transform()."""
    result = prep_pipe.transform(test_df)
    assert type(result) == pd.DataFrame


def test_df_after_ohe_transform():
    """Checks transformation via ohe transformer."""
    result = prep_pipe.transform(test_df)
    assert "num_col" in result.columns
    for i in range(1, 10):
        assert f"cat_col1_{i}" in result.columns
        assert f"cat_col2_{i}{i}" in result.columns


def test_dtype_of_ohe_transform():
    """Checks for dtype uint8 of ohe transformation."""
    result = prep_pipe.transform(test_df).select_dtypes(include=("uint8")).columns
    assert all("cat_col" in col for col in result)
