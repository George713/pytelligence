import pandas as pd

import pycarrot as pc

# Preparation
df_clf = pd.DataFrame(
    data={
        "num_col": [i for i in range(100)],
        "cat_col": [str(i) for i in range(100)],
        "target_clf": [0] * 50 + [1] * 50,
    }
)

config = pc.init_config("./tests/config_test.yml")

setup, _, _ = pc.modelling.prepare_data(
    train_data=df_clf,
    config=config,
)
# Preparation End

compare_df, algo_list, model_list = pc.modelling.tune_hyperparams(
    setup=setup,
    include=["lr", "knn"],
    optimize="f1",
    n_trials=50,
    return_models=True,
)


def test_return_types():
    """Tests return types of tune_hyperparams()."""
    assert type(compare_df) == pd.DataFrame
    assert type(algo_list) == list
    assert type(model_list) == list


def test_return_type_objective_function():
    """Tests return type of `_get_objective_function()`."""
    objective_fn = pc.modelling._tune_hyperparams._get_objective_function(
        algorithm="lr", optimize="f1"
    )
    assert callable(objective_fn)
