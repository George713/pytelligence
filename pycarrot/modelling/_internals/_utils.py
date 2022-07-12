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


def check_include(include: List[str]):
    """Checks `include` list for correct strings.

    Parameters
    ----------
    include : List[str]
        List of strings referring to implemented model algorithms.

    Raises
    ------
    LookupError
        In case a string is not matched by an implemented algorithm,
        raises a LookupError.
    """
    available_algos = get_available_algos()
    for entry in include:
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
