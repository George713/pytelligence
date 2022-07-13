from sklearn.linear_model import LogisticRegression
import optuna
import pycarrot as pc


default_model = pc.modelling._internals.get_model_instance(algorithm="lr")

study = optuna.create_study(direction="maximize")
trial = study.ask()
adjusted_model = pc.modelling._internals.get_model_instance(algorithm="lr", trial=trial)

manually_adj_model = pc.modelling._internals.get_model_instance(
    algorithm="lr", hyperparams={"l1_ratio": 0.2}
)


def test_return_type_get_model_instance():
    """Checks return type of get_model_instance()."""
    assert type(default_model) == LogisticRegression


def test_default_hyperparams():
    default_params = default_model.get_params()
    assert default_params["solver"] == "saga"
    assert default_params["max_iter"] == 1000
    assert default_params["l1_ratio"] == 1


def test_adjusted_hyperparams():
    """
    Checks whether the `C` hyperparameter has been adjusted.
    Its default value is 1 and hitting exactly that is very unlikely.
    This serves as a check whether all hyperparameters selected for
    tuning have been adjusted, although not all are checked for.
    (This is due to the high likelihood of the categorical
    hyperparameter `penalty` not passing the test.)
    """
    adjusted_params = adjusted_model.get_params()
    assert adjusted_params["C"] != 1


def test_manually_adjusted_hyperparams():
    manually_adj_params = manually_adj_model.get_params()
    assert manually_adj_params["l1_ratio"] == 0.2
