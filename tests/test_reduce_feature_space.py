from sklearn.datasets import load_iris
import pandas as pd

import pycarrot as pc

data = load_iris(as_frame=True)
df_clf = pd.concat([data.data, data.target], axis=1)
df_clf = df_clf.query("target < 2").reset_index(drop=True)

config = {
    "modelling": {
        "target_clf": "target",
        "numeric_cols": [
            "sepal width (cm)",
            "sepal length (cm)",
            "petal width (cm)",
            "petal length (cm)",
        ],
        "categorical_cols": [],
        "normalization": False,
    }
}

prepare_data_result = pc.modelling.prepare_data(
    train_data=df_clf,
    config=config,
)
setup, _, _ = prepare_data_result

result = pc.modelling.reduce_feature_space(
    setup,
    algorithm="perceptron",
    metric="accuracy",
    reference_metric=1.0,
    acceptable_loss=0.6,
)


def test_fn_exists():
    assert result is not None


def test_return_types():
    print(result)
    assert type(result) == list
