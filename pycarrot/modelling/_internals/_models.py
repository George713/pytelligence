"""
Container for implemented models.

Each implemented model is described by a class containing attributes with
  1) the base model to use when comparing algorithms
  2) the hyperparameter space to search when tuning

When a model is required, it can be requested using the function
`_get_base_model(type="classification", "lr")`.
"""

from sklearn.linear_model import LogisticRegression
from optuna import trial


class ModelContainer:
    self.base_model
    self.hyper_params

    def get_model_with_hyperparams():
        pass


class ClfLinearRegression(ModelContainer):
    def __init__(self):
        self.base_model = LogisticRegression(solver="saga", max_iter=1000)
        self.hyperparams = {
            "c": trial.suggest_float("c", 1e-6, 1e2, log=True),
            "penalty": trial.suggest_categorical(
                "penalty", ["l1", "l2", "elasticnet", "none"]
            ),
        }
