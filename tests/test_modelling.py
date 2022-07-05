import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import pycarrot as pc
from pycarrot.modelling._setup import Setup

config = pc.init_config("./config_test.yml")

df_clf = pd.DataFrame(
    data={
        "num_col": [i for i in range(100)],
        "target_clf": [0] * 50 + [1] * 50,
    }
)

prepare_data_result = pc.modelling.prepare_data(
    train_data=df_clf,
    config=config,
)
setup, _, _ = prepare_data_result

compare_algorithms_result = pc.modelling.compare_algorithms(
    setup,
)
compare_df, model_dict = compare_algorithms_result

available_algo_result = (
    pc.modelling._compare_algorithms._get_available_algos()
)

unfitted_model = (
    pc.modelling._train_model._get_unfitted_model("lr")
)

X_train, y_train = setup.X_train, setup.y_clf_train

cv_results = cross_validate(
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

result_agg_metrics = (
    pc.modelling._train_model._aggregate_metrics(
        cv_results, "lr"
    )
)


def test_return():
    assert type(prepare_data_result) == tuple
    assert type(compare_algorithms_result) == tuple


def test_return_setup():
    assert type(prepare_data_result[0]) == Setup
    assert type(prepare_data_result[1]) == pd.DataFrame
    assert type(prepare_data_result[2]) == pd.Series


def test_return_setup_X_train():
    assert hasattr(prepare_data_result[0], "X_train")


def test_return_setup_X_train_value():
    assert (
        type(prepare_data_result[0].X_train) == pd.DataFrame
    )


def test_return_setup_y_train():
    assert hasattr(prepare_data_result[0], "y_clf_train")


def test_return_setup_y_train_value():
    assert (
        type(prepare_data_result[0].y_clf_train)
        == pd.Series
    )


def test_return_compare():
    assert (
        type(compare_algorithms_result[0]) == pd.DataFrame
    )
    assert type(compare_algorithms_result[1]) == dict


def test_get_available_algos_type():
    assert type(available_algo_result) == list
    assert all(
        type(algo) == str for algo in available_algo_result
    )
    assert all(
        algo in available_algo_result
        for algo in [
            "lr",
            "dt",
            "extratree",
            "extratrees",
            "rf",
            "ridge",
            "perceptron",
            "passive-aggressive",
            "knn",
            "nb",
            "linearsvc",
            "rbfsvc",
        ]
    )


def test_get_available_algos_type_of_entries():
    assert all(
        type(algo) == str for algo in available_algo_result
    )


def test_get_available_algos_entries():
    assert all(
        algo in available_algo_result for algo in ["lr"]
    )


def test_check_include_correct():
    result = (
        pc.modelling._compare_algorithms._check_include(
            available_algo_result
        )
    )
    assert result is None


def test_check_include_invalid():
    try:
        result = (
            pc.modelling._compare_algorithms._check_include(
                ["lr", "wrong_algo_name", 5]
            )
        )
    except Exception as e:
        result = e
    assert type(result) == LookupError


def test_train_model_return_type():
    result = pc.modelling._train_model.train_model(
        "lr", setup, return_models=True
    )
    assert type(result) == tuple
    assert type(result[0]) == LogisticRegression


def test_get_unfitted_model():
    assert type(unfitted_model) == LogisticRegression


def test_get_unfitted_model_error():
    try:
        result = (
            pc.modelling._train_model._get_unfitted_model(
                "unknown_algo"
            )
        )
    except Exception as e:
        result = e
    assert type(result) == LookupError


def test_agg_metrics_type():
    assert type(result_agg_metrics) == pd.DataFrame


def test_agg_metrics_cols():
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


def test_check_sort():
    result = pc.modelling._compare_algorithms._check_sort(
        None
    )
    assert result is None


def test_check_sort_fail():
    try:
        result = (
            pc.modelling._compare_algorithms._check_sort(
                "f7"
            )
        )
    except Exception as e:
        result = e
    assert type(result) == LookupError
