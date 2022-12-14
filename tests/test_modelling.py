import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import pytelligence as pt


@pytest.fixture
def df_clf():
    return pd.DataFrame(
        data={
            "num_col": [i for i in range(100)],
            "cat_col": [str(i) for i in range(100)],
            "target_clf": [0] * 50 + [1] * 50,
        }
    )


@pytest.fixture
def prepare_data_result(df_clf):
    return pt.modelling.prepare_data(
        train_data=df_clf,
        config_path="./tests/config_test.yml",
    )


@pytest.fixture
def setup(prepare_data_result):
    return prepare_data_result[0]


@pytest.fixture
def compare_algorithms_result(setup):
    return pt.modelling.compare_algorithms(setup)


@pytest.fixture
def compare_df(compare_algorithms_result):
    return compare_algorithms_result[0]


@pytest.fixture
def algo_list(compare_algorithms_result):
    return compare_algorithms_result[1]


@pytest.fixture
def model_list(compare_algorithms_result):
    return compare_algorithms_result[2]


@pytest.fixture
def unfitted_model():
    return pt.modelling._internals.get_model_instance("lr")


@pytest.fixture
def X_train(setup):
    return setup.X_train


@pytest.fixture
def y_train(setup):
    return setup.y_clf_train


@pytest.fixture
def cv_results(unfitted_model, X_train, y_train):
    return cross_validate(
        unfitted_model,
        X_train,
        y_train,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
        n_jobs=-1,
    )


@pytest.fixture
def result_agg_metrics(cv_results):
    return pt.modelling._internals.aggregate_metrics(cv_results, "lr")


def test_return(prepare_data_result, compare_algorithms_result):
    assert type(prepare_data_result) == tuple
    assert type(compare_algorithms_result) == tuple


def test_return_setup(prepare_data_result):
    assert type(prepare_data_result[0]) == pt.modelling._internals.Setup
    assert type(prepare_data_result[1]) == pd.DataFrame
    assert type(prepare_data_result[2]) == pd.Series


def test_return_setup_X_train(prepare_data_result):
    assert hasattr(prepare_data_result[0], "X_train")


def test_return_setup_X_train_value(prepare_data_result):
    assert type(prepare_data_result[0].X_train) == pd.DataFrame


def test_return_setup_y_train(prepare_data_result):
    assert hasattr(prepare_data_result[0], "y_clf_train")


def test_return_setup_y_train_value(prepare_data_result):
    assert type(prepare_data_result[0].y_clf_train) == pd.Series


def test_return_compare(compare_algorithms_result):
    assert type(compare_algorithms_result[0]) == pd.DataFrame
    assert type(compare_algorithms_result[1]) == list
    assert type(compare_algorithms_result[2]) == list


def test_train_model_return_type(setup):
    result = pt.modelling._train_model.train_model("lr", setup, return_models=True)
    assert type(result) == tuple
    assert type(result[0]) == LogisticRegression


def test_get_unfitted_model(unfitted_model):
    assert type(unfitted_model) == LogisticRegression


def test_agg_metrics_type(result_agg_metrics):
    assert type(result_agg_metrics) == pd.DataFrame


def test_agg_metrics_cols(result_agg_metrics):
    assert all(
        col in result_agg_metrics
        for col in [
            "algorithm",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
    )
