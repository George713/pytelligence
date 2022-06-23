from typing import Tuple, List

from ._train_model import train_model


def reduce_feature_space(
    setup: dict,
    algorithm: str,
    metric: str,
    reference_metric: float,
    acceptable_loss: float,
):
    """
    Reduces the feature space used for model training
    by iteratively removing the feature with the smallest
    impact on model performance.

    Given an algorithm and metric, retrains models with
    one feature removed. Notes the metric when the weakest
    feature is removed and compares it with the acceptable
    loss threshold. If the threshold has not been reached
    the process is repeated.

    If a metric higher than the reference_metric is found,
    it is printed.

    Arguments
    ---------
    setup : dict
        Dictionary containing the prepared data and further
        configurations.

    algorithm : str
        Algorithm to use for model training.

    metric : str
        Metric to use for evaluating model performance.

    reference_metric : float
        Metric value to use as baseline.

    acceptable_loss : float
        Acceptable loss threshold, e.g. 0.95. Once a value
        of acceptable_loss * reference_metric is undercut,
        the feature space reduction is reduced.

    Returns
    -------
    best_feature_list : List[str]
    """
    # Initiate reference values
    threshold = acceptable_loss * reference_metric
    feature_list = setup["X_train"].columns.to_list()
    best_feature_list = feature_list[:]
    new_metric = reference_metric

    # Iteratively remove features
    while (threshold <= new_metric) & (
        len(feature_list) > 1
    ):
        (worst_feature, new_metric) = _find_worst_feature(
            setup,
            algorithm,
            metric,
            feature_list,
        )
        feature_list.remove(worst_feature)
        print(
            f"New metric: {new_metric:.3}, worst feature: {worst_feature}"
        )

        # Update reference_metric and threshold if
        # reference_metric was improved upon
        if new_metric > reference_metric:
            reference_metric = new_metric
            threshold = acceptable_loss * reference_metric
            print(f"Updating reference metric...")

        # Update best_feature_list if metric is equal to
        # reference_metric
        if new_metric == reference_metric:
            best_feature_list = feature_list[:]

    return best_feature_list


def _find_worst_feature(
    setup: dict,
    algorithm: str,
    metric: str,
    feature_list: List[str],
) -> Tuple[str, float]:
    """
    Finds worst feature given a specific algorithm and
    metric.

    Loops over all features in feature_list, removes the
    individual feature and trains a models instance on the
    remaining features. Afterwards the feature impact is
    evaluated by calculating the achieved metric.

    The feature associated with the largest metric after
    removal of the feature is returned alongside the metric.

    Arguments
    ---------
    setup : dict
        Dictionary containing the prepared data and further
        configurations.

    algorithm : str
        Algorithm to use for model training.

    metric : str
        Metric to use for evaluating model performance.

    Returns
    -------
    worst_feature: str
        Name of the feature best to be left out for training.

    new_metric: float
        Metric value achieved when leaving out worst feature.
    """
    # Initiate result list
    removed_feature = []
    new_metric = []

    # Iterate over feature list
    for feature in feature_list:
        _, metrics = train_model(
            algorithm,
            setup,
            feature_list=[
                feat
                for feat in feature_list
                if feat != feature
            ],
        )
        removed_feature.append(feature)
        new_metric.append(metrics[metric].values[0])

    # Retrieve worst feature and metric
    new_metric_max = max(new_metric)
    max_index = new_metric.index(new_metric_max)
    worst_feature = removed_feature[max_index]

    return worst_feature, new_metric_max
