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


@pytest.fixture
def tmp_file(tmp_path, today):
    new_model = tmp_path / f"model_{today}_lr_#16.joblib"
    new_model.write_text("arb. text")


def test_combine_pipeline_and_model():
    full_pipe = _combine_pipeline_and_model(
        prep_pipe=Pipeline(steps=[]),
        model=LogisticRegression(),
        y_clf_encoder=None,
    )
    assert full_pipe.steps[-1][0] == "model_wrapper"
    assert type(full_pipe.steps[-1][1]) == pt.modelling._export_model.ModelWrapper
    assert type(full_pipe.steps[-1][1].estimator) == LogisticRegression


def test_get_algo_abbreviation():
    result = _get_algo_abbreviation(model_type=type(LogisticRegression()))
    assert result == "lr"


def test_get_new_number_with_preexisting_file(tmp_path, tmp_file, today):
    result = _get_new_number(target_dir=tmp_path, temp_name=f"model_{today}_lr")
    assert result == 17


def test_get_new_number_without_preexisting_file(tmp_path, today):
    result = _get_new_number(target_dir=tmp_path, temp_name=f"model_{today}_xgb")
    assert result is None


def test_get_export_name_with_preexisting_file(tmp_path, tmp_file, today):
    result = _get_export_name(target_dir=tmp_path, model=LogisticRegression())
    assert result == f"model_{today}_lr_#17.joblib"


def test_get_export_name_without_preexisting_file(tmp_path, today):
    result = _get_export_name(target_dir=tmp_path, model=GaussianNB())
    assert result == f"model_{today}_nb_#1.joblib"


def test_export_model(tmp_path, today):
    # Create file
    setup = pt.modelling._internals.Setup(
        X_train=None,
        y_clf_train=None,
        y_clf_encoder=None,
        feature_scaling=False,
        prep_pipe=Pipeline(steps=[]),
    )
    model = GaussianNB()
    export_model(setup=setup, model=model, target_dir=tmp_path)

    # Verify file was created
    assert Path(tmp_path / f"model_{today}_nb_#1.joblib").exists()
