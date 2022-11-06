"""
Tests for pycarrot.modelling.export_model
"""

import datetime
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

import pytelligence as pt
from pytelligence.modelling import export_model
from pytelligence.modelling._export_model import (
    _combine_pipeline_and_model,
    _get_algo_abbreviation,
    _get_export_name,
    _get_new_number,
)


@pytest.fixture
def today():
    return datetime.date.today()


test_file = [file for file in Path("./tests/export_test/").glob("./*")][0]
test_file.rename(test_file.parent / f"model_{datetime.date.today()}_lr_#16.joblib")
# End of Preparation


def test_combine_pipeline_and_model():
    full_pipe = _combine_pipeline_and_model(
        prep_pipe=Pipeline(steps=[]),
        model=LogisticRegression(),
    )
    assert full_pipe.steps[-1][0] == "model"
    assert type(full_pipe.steps[-1][1]) == LogisticRegression


def test_get_algo_abbreviation():
    result = _get_algo_abbreviation(model_type=type(LogisticRegression()))
    assert result == "lr"


def test_get_new_number_with_preexisting_file(today):
    result = _get_new_number(
        target_dir="./tests/export_test/", temp_name=f"model_{today}_lr"
    )
    assert result == 17


def test_get_new_number_without_preexisting_file(today):
    result = _get_new_number(
        target_dir="./tests/export_test/", temp_name=f"model_{today}_xgb"
    )
    assert result is None


def test_get_export_name_with_preexisting_file(today):

    result = _get_export_name(
        target_dir="./tests/export_test/", model=LogisticRegression()
    )
    assert result == f"model_{today}_lr_#17.joblib"


def test_get_export_name_without_preexisting_file(today):
    result = _get_export_name(target_dir="./tests/export_test/", model=GaussianNB())
    assert result == f"model_{today}_nb_#1.joblib"


def test_export_model(today):
    # Verify that file does not exist yet
    assert Path(f"./tests/export_test/model_{today}_nb_#1.joblib").exists() is False

    # Create file
    setup = pt.modelling._internals.Setup(
        X_train=None,
        y_clf_train=None,
        y_clf_encoder=None,
        feature_scaling=False,
        prep_pipe=Pipeline(steps=[]),
    )
    model = GaussianNB()
    target_dir = "./tests/export_test/"
    export_model(setup=setup, model=model, target_dir=target_dir)

    # Verify file was created
    assert Path(f"./tests/export_test/model_{today}_nb_#1.joblib").exists()

    # Delete file again
    Path(f"./tests/export_test/model_{today}_nb_#1.joblib").unlink()
