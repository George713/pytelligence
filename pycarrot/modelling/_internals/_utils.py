"""
Contains utilty functionality used by various modules.
"""

from typing import List, Optional


def get_available_algos() -> List[str]:
    """
    Returns a list of strings where each string is the
    abbreviation of an algorithm.
    """
    return [
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


def check_include(algo_list: List[str]):
    """Checks `algo_list` for correct strings.

    Parameters
    ----------
    algo_list : List[str]
        List of strings referring to implemented model algorithms.

    Raises
    ------
    LookupError
        In case a string is not matched by an implemented algorithm,
        raises a LookupError.
    """
    available_algos = get_available_algos()
    for entry in algo_list:
        if entry not in available_algos:
            raise LookupError(
                f"'{entry}' was provided in the include parameter, but is not among the avaiable algorithms."
            )


def check_metric(metric: Optional[str]):
    """Checks whether `metric` is among the implemented metrics.
    Raises LookupError if `metric` is not implemented and not None.

    Raises
    ------
    LookupError
        In case `metric` is not matched by an implemented algorithm
        or None, raises a LookupError.
    """
    if metric not in [
        None,
        "algorithm",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        raise LookupError(
            f"'{metric}' was provided as sort parameter, but is not among the avaiable metrics."
        )


def check_normalization(algo_list: List[str], normalization: bool) -> None:
    """Checks for normalization and writes note to logger.INFO if any of
    the utilized algorithms would profit from normalization.

    Parameters
    ----------
    algo_list : List[str]
        List of utilized algorithms.

    normalization : bool
        Option documented in setup object. Specified in config.
    """
    if not normalization:
        affected_algos = [algo for algo in algo_list if algo in ["lr"]]
        print(
            f"The algorithms {affected_algos} work suboptimally without normalized data. Consider turning it on within the config."
        )
