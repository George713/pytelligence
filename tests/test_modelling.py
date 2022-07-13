import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import pycarrot as pc

df_clf = pd.DataFrame(
    data={
        "num_col": [i for i in range(100)],
        "cat_col": [str(i) for i in range(100)],
        "target_clf": [0] * 50 + [1] * 50,
    }
)

prepare_data_result = pc.modelling.prepare_data(
    train_data=df_clf,
    config_path="./tests/config_test.yml",
)
setup, _, _ = prepare_data_result

compare_algorithms_result = pc.modelling.compare_algorithms(setup)
compare_df, algo_list, model_list = compare_algorithms_result


unfitted_model = pc.modelling._internals.get_model_instance("lr")

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

result_agg_metrics = pc.modelling._train_model._aggregate_metrics(cv_results, "lr")


def test_return():
    assert type(prepare_data_result) == tuple
    assert type(compare_algorithms_result) == tuple


def test_return_setup():
    assert type(prepare_data_result[0]) == pc.modelling._internals.Setup
    assert type(prepare_data_result[1]) == pd.DataFrame
    assert type(prepare_data_result[2]) == pd.Series


def test_return_setup_X_train():
    assert hasattr(prepare_data_result[0], "X_train")


def test_return_setup_X_train_value():
    assert type(prepare_data_result[0].X_train) == pd.DataFrame


def test_return_setup_y_train():
    assert hasattr(prepare_data_result[0], "y_clf_train")


def test_return_setup_y_train_value():
    assert type(prepare_data_result[0].y_clf_train) == pd.Series


def test_return_compare():
    assert type(compare_algorithms_result[0]) == pd.DataFrame
    assert type(compare_algorithms_result[1]) == list
    assert type(compare_algorithms_result[2]) == list


def test_train_model_return_type():
    result = pc.modelling._train_model.train_model("lr", setup, return_models=True)
    assert type(result) == tuple
    assert type(result[0]) == LogisticRegression


def test_get_unfitted_model():
    assert type(unfitted_model) == LogisticRegression


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
