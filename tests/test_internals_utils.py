import pytest

import pytelligence as pt


@pytest.fixture
def available_algo_result():
    return pt.modelling._internals.get_available_algos()


def test_get_available_algos_type(available_algo_result):
    assert type(available_algo_result) == list
    assert all(type(algo) == str for algo in available_algo_result)
    assert all(
        algo in available_algo_result
        for algo in [
            "lr",
            # "dt",
            # "extratree",
            # "extratrees",
            # "rf",
            # "ridge",
            # "perceptron",
            # "passive-aggressive",
            # "knn",
            "nb",
            # "linearsvc",
            # "rbfsvc",
        ]
    )


def test_get_available_algos_type_of_entries(available_algo_result):
    assert all(type(algo) == str for algo in available_algo_result)


def test_get_available_algos_entries(available_algo_result):
    assert all(algo in available_algo_result for algo in ["lr"])


def testc_heck_include_correct(available_algo_result):
    result = pt.modelling._internals.check_include(available_algo_result)
    assert result is None


def test_check_include_invalid():
    try:
        result = pt.modelling._internals.check_include(["lr", "wrong_algo_name", 5])
    except Exception as e:
        result = e
    assert type(result) == LookupError


def test_check_metric():
    result = pt.modelling._internals.check_metric(None)
    assert result is None


def test_check_metric_fail():
    try:
        result = pt.modelling._internals.check_metric("f7")
    except Exception as e:
        result = e
    assert type(result) == LookupError
