import pandas as pd
from sklearn.datasets import load_iris

import pytelligence as pt

data = load_iris(as_frame=True)
df_clf = pd.concat([data.data, data.target], axis=1)
df_clf = df_clf.query("target < 2").reset_index(drop=True)

prepare_data_result = pt.modelling.prepare_data(
    train_data=df_clf,
    config_path="./tests/config_iris.yml",
)
setup, _, _ = prepare_data_result

result = pt.modelling.reduce_feature_space(
    setup,
    algorithm="lr",
    metric="accuracy",
    reference_metric=1.0,
    acceptable_loss=0.6,
)

result_w_hyperparams = pt.modelling.reduce_feature_space(
    setup,
    algorithm="lr",
    metric="accuracy",
    reference_metric=1.0,
    acceptable_loss=0.6,
    hyperparams={"l1_ratio": 0.2},
)


def test_fn_exists():
    assert result is not None


def test_return_types():
    assert type(result) == list


def test_return_types():
    assert type(result_w_hyperparams) == list
