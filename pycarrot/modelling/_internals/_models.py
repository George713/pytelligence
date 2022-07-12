"""
Container for implemented models.

Each implemented model is described by a class containing attributes with
  1) the base model to use when comparing algorithms
  2) the hyperparameter space to search when tuning

When a model is required, it can be requested using the function
`_get_base_model(type="classification", "lr")`.
"""

from abc import ABC, abstractmethod
from typing import Optional
from sklearn.linear_model import LogisticRegression
import optuna


def get_model_instance(algorithm: str, trial: Optional[optuna.Trial] = None):
    """Returns model instance for `algorithm`.

    When a `trial` instance is provided, the hyperparams of the
    model instance are set using optuna's trial object.

    Parameters
    ----------
    algorithm : str
        Algorithm specifying what type of model to return.

    trial : Optional[optuna.Trial]
        Optuna's trial object used to specify hyperparameters to use.

    Returns
    -------
    model_instance
        A model instance based on `algorithm` with base or adjusted
        hyperparameters depending on a trial-object being provided.
    """
    if algorithm == "lr":
        return ClfLinearRegression(trial=trial).get_model()


class ModelContainer(ABC):
    """
    Abstract class for implementing algorithms into pycarrot.

    Each child class must implement two attributes:
      1) self.base_model
      2) self.hyperparams

    `base_model` is a model instance with default hyperparameters.
    "Default" in this context typically refers to the library's standard
    hyperparameters, sometimes with slight adjustments for the target audience
    of pycarrot, e.g. max_iter for LinearRegression and larger datasets.

    `hyperparams` is either None or a dictionary, if the child class is
    instaniated with a trial object. Keys of that dictionary must mirrow
    the model's hyperparameter names and the values consist of optuna's
    trial suggestions. See example below.

    This parent class also provides a generic get_model function that
    returns a model instance based on whether a trial object has been
    provided or not.

    Example
    -------
    >>> self.base_model = LogisticRegression(solver="saga", max_iter=1000)
    >>> self.hyperparams = (
            {
                "C": trial.suggest_float("c", 1e-6, 1e2, log=True),
                "penalty": trial.suggest_categorical(
                    "penalty", ["l1", "l2", "elasticnet", "none"]
                ),
            }
            if trial
            else None
        )
    """

    @abstractmethod
    def __init__(self, trial: Optional[optuna.Trial] = None):
        self.base_model = None
        self.hyperparams = None

    def get_model(self):
        """Returns model instance based on child class attributes."""
        return (
            self.base_model
            if self.hyperparams is None
            else self.base_model.set_params(**self.hyperparams)
        )


class ClfLinearRegression(ModelContainer):
    """
    ModelContainer for classification algorithm `LinearRegression`.

    Hyperparameter choice:
      "C": Inverse of regularization strength. In general strong impact.
      "penalty": Norm of the penalty. Typically strong impact.
      "solver": Algorithm for optimization set to `saga` as it supports
         all penalty norms and is typically fastest on large datasets.
         Requires normalized features for good convergence. Typically
         only minor differences in model quality are observed by varying
         the solver.
    """

    def __init__(self, trial: Optional[optuna.Trial] = None):
        self.base_model = LogisticRegression(solver="saga", max_iter=1000)
        self.hyperparams = (
            {
                "C": trial.suggest_float("c", 1e-6, 1e2, log=True),
                "penalty": trial.suggest_categorical(
                    "penalty", ["l1", "l2", "elasticnet", "none"]
                ),
            }
            if trial
            else None
        )
