import numpy as np
import optuna
import pandas as pd
import plotly
from sklearn.linear_model import LogisticRegression

import pytelligence as pt

# Preparation
df_clf = pd.DataFrame(
    data={
        "num_col": [i for i in range(100)],
        "cat_col": [str(i) for i in range(100)],
        "target_clf": [0] * 50 + [1] * 50,
    }
)

setup, _, _ = pt.modelling.prepare_data(
    train_data=df_clf,
    config_path="./tests/config_test.yml",
)

objective_fn = pt.modelling._tune_hyperparams._get_objective_function(
    algorithm="lr", optimize="f1", X=setup.X_train, y=setup.y_clf_train
)

study = optuna.create_study(direction="maximize")
trial = study.ask()
# Preparation End

compare_df, model_list, opt_history_dict = pt.modelling.tune_hyperparams(
    setup=setup,
    include=["lr"],
    optimize="f1",
    n_trials=2,
    return_models=True,
)


def test_return_types():
    """Tests return types of tune_hyperparams()."""
    assert type(compare_df) == pd.DataFrame
    assert type(model_list) == list
    assert type(opt_history_dict) == dict


def test_return_type_objective_function():
    """Tests return type of `_get_objective_function()`."""
    assert callable(objective_fn)


def test_return_type_of_objective_function():
    """Tests return type of `objective_fc`."""
    result = objective_fn(trial)
    assert type(result) == np.float64


def test_columns_of_returned_compare_df():
    assert set(compare_df.columns) == set(
        [
            "algorithm",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "Fit time (s)",
            "hyperparams",
        ]
    )


def test_type_of_1st_entry_in_model_list():
    assert type(model_list[0]) == LogisticRegression


def test_value_type_in_opt_history_dict():
    assert type(opt_history_dict["lr"]) == plotly.graph_objs.Figure
