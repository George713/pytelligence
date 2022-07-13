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
from sklearn.naive_bayes import GaussianNB


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
        algo = ClfLinearRegression
    elif algorithm == "nb":
        algo = ClfGaussianNB

    return algo(trial=trial).get_model()


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

    `tuning_params` is either None or a dictionary if the child class is
    instaniated with a trial object. Keys of that dictionary must mirrow
    the model's hyperparameter names and the values consist of optuna's
    trial suggestions. See example below.

    This parent class also provides a generic get_model function that
    returns a model instance based on whether a trial object has been
    provided or not.

    Example
    -------
    >>> self.base_model = LogisticRegression(
            solver="saga", max_iter=1000, penalty="elasticnet", l1_ratio=1
        )
    >>> self.tuning_params = (
            {
                "C": trial.suggest_float("C", 1e-6, 1e2, log=True),
                "penalty": trial.suggest_categorical(
                    "penalty", ["elasticnet", "none"]
                ),
                "l1_ratio": trial.suggest_uniform("l1_ratio", 0, 1),
            }
            if trial
            else None
        )
    """

    @abstractmethod
    def __init__(self, trial: Optional[optuna.Trial] = None):
        self.base_model = None
        self.tuning_params = None

    def get_model(self):
        """Returns model instance based on child class attributes."""
        return (
            self.base_model
            if self.tuning_params is None
            else self.base_model.set_params(**self.tuning_params)
        )


class ClfLinearRegression(ModelContainer):
    """
    ModelContainer for classification algorithm `LinearRegression`.

    See: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Hyperparameter choice:
      "solver": Algorithm for optimization set to `saga` as it supports
         all penalty norms and is typically fastest on large datasets.
         Requires normalized features for good convergence. Typically
         only minor differences in model quality are observed by varying
         the solver.
      "penalty": Norm of the penalty. Typically strong impact. Only includes
         `elasticnet` as `l1` and `l2` are special cases covered by l1_ratio.
         Forgoing the penalty (penalty=`none`) is covered by very large values
         of `C`.
      "C": Inverse of regularization strength. In general strong impact.
      "l1_ratio": If solver `elasticnet` is used, the l1_ratio determines
         the combination ratio between l1 and l2. 0 corresponds to pure `l1`
         while a value of 1 results in pure `l2`.
    """

    def __init__(self, trial: Optional[optuna.Trial] = None):
        self.base_model = LogisticRegression(
            solver="saga", max_iter=1000, penalty="elasticnet", l1_ratio=1
        )
        self.tuning_params = (
            {
                "C": trial.suggest_float("C", 1e-6, 1e2, log=True),
                "l1_ratio": trial.suggest_uniform("l1_ratio", 0, 1),
            }
            if trial
            else None
        )


class ClfGaussianNB(ModelContainer):
    """
    ModelContainer for classification algorithm `GaussianNB`.

    See: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

    Hyperparameter choice:
      "var_smoothing": Portion of the largest variance of all features
      that is added to variances for calculation stability. Only tunable
      feature.
    """

    def __init__(self, trial: Optional[optuna.Trial] = None):
        self.base_model = GaussianNB()
        self.tuning_params = (
            {
                "var_smoothing": trial.suggest_float(
                    "var_smoothing", 1e-12, 1e0, log=True
                ),
            }
            if trial
            else None
        )
