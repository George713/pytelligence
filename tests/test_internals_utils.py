import pycarrot as pc

available_algo_result = pc.modelling._internals.get_available_algos()


def test_get_available_algos_type():
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


def test_get_available_algos_type_of_entries():
    assert all(type(algo) == str for algo in available_algo_result)


def test_get_available_algos_entries():
    assert all(algo in available_algo_result for algo in ["lr"])


def testc_heck_include_correct():
    result = pc.modelling._internals.check_include(available_algo_result)
    assert result is None


def test_check_include_invalid():
    try:
        result = pc.modelling._internals.check_include(["lr", "wrong_algo_name", 5])
    except Exception as e:
        result = e
    assert type(result) == LookupError


def test_check_metric():
    result = pc.modelling._internals.check_metric(None)
    assert result is None


def test_check_metric_fail():
    try:
        result = pc.modelling._internals.check_metric("f7")
    except Exception as e:
        result = e
    assert type(result) == LookupError
