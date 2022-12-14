"""
Logic related to exporting a trained model with a corresponding
preprocessing pipeline.
"""
import abc
import datetime
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from . import _internals

logger = logging.getLogger(f"stream.{__name__}")


def export_model(setup: _internals.Setup, model, target_dir: str) -> None:
    """Exports provided `model` with preprocessing pipeline stored in
    setup.

    The output file is binarized using joblib and saved to `target_dir`.
    The output filename follows the syntax
      `model_{date}_{algo}_#{number}.joblib`
    where
    - `date` is the current date in format YYYY-MM-DD
    - `algo` is the utilised model algorithm, and
    - `number` is a number assigned when other models with the same
    name already exist.

    Example
    -------
    >>> pt.modelling.export_model(
            setup=setup,
            model=model_list[0],
            target_dir="./",
        )

    Parameters
    ----------
    setup : _internals.Setup
        Setup object containing preprocessing pipeline.

    model :
        Trained model instance.

    target_dir : str
        Target directory for saving the output file to.
    """
    full_pipe = _combine_pipeline_and_model(
        prep_pipe=setup.prep_pipe, model=model, y_clf_encoder=setup.y_clf_encoder
    )
    export_name = _get_export_name(target_dir=target_dir, model=model)
    joblib.dump(full_pipe, Path(target_dir) / export_name)
    logger.info(f"Exported modelling pipeline to '{Path(target_dir) / export_name}'")


def _combine_pipeline_and_model(
    prep_pipe: Pipeline, model, y_clf_encoder: LabelEncoder
) -> Pipeline:
    """Adds `model` as last step to `prep_pipe`."""
    # Making deep copy, so that the model-step is not added to the
    # original pipeline object.
    prep_pipe_copy = deepcopy(prep_pipe)
    prep_pipe_copy.steps.append(
        ("model_wrapper", ModelWrapper(estimator=model, y_clf_encoder=y_clf_encoder))
    )
    # prep_pipe_copy.steps.append(("model", model))
    return prep_pipe_copy


def _get_export_name(target_dir: str, model) -> str:
    """Generates filename for `full_pipe`."""
    date = str(datetime.date.today())
    algo = _get_algo_abbreviation(model_type=type(model))
    temp_name = f"model_{date}_{algo}_#"
    new_number = _get_new_number(target_dir=target_dir, temp_name=temp_name)
    return (
        f"model_{date}_{algo}_#{new_number}.joblib"
        if new_number
        else f"model_{date}_{algo}_#1.joblib"
    )


def _get_algo_abbreviation(model_type: abc.ABCMeta) -> str:
    """Returns string abbreviation for provided algorithm."""
    algo_dict = _internals.get_algo_dict()
    return [key for key in algo_dict.keys() if algo_dict[key] == model_type][0]


def _get_new_number(target_dir: str, temp_name: str) -> Optional[int]:
    """Determines the number value for the new file name by
    checking `target_dir` for pre-existing files with matching
    name pattern.

    Parameters
    ----------
    target_dir : str
        Target directory for new file. This is also where for
        pre-existing files of similar name is checked.

    temp_name : str
        Basic name structure for new file.

    Returns
    -------
    Optional[int]
        Integer value of the new number, or None.
        None is returned when no pre-existing file is found.
    """
    files_in_target_dir = [
        str(path.name) for path in Path(target_dir).glob("./*") if path.is_file()
    ]
    files_matching_pattern = [
        filename for filename in files_in_target_dir if temp_name in filename
    ]
    if len(files_matching_pattern) > 0:
        last_entry = sorted(files_matching_pattern)[-1]
        last_number = int(last_entry.rsplit("#")[-1].split(".joblib")[0])
        return last_number + 1
    else:
        return None


class ModelWrapper:
    """Wrapper class for estimator and label encoder.

    In case of a classification task the target variable is often
    encoded as string format. During preprocessing such target columns
    are encoded to numerical values so that the subsequent estimator can
    handle the data. This wrapper class facilitates the decoding of the
    estimator's prediction into the orignal format.

    Attributes
    ----------
    estimator : Estimator in scikit-learn format
        Estimator trained during a previous step.
    y_clf_encoder : LabelEncoder
        Retrieved from Setup object, i.e. setup.y_clf_encoder

    Methods
    -------
    predict(X)
        Calls the predict()-method of the estimator and applies decoding
        of the label encoder if one was used.
    predict_proba(X)
        Calls the estimator's predict_proba()-method.
    """

    def __init__(self, estimator, y_clf_encoder: Union[LabelEncoder, None]) -> None:
        self.estimator = estimator
        self.y_clf_encoder = y_clf_encoder

    def predict(self, X) -> np.ndarray:
        if self.y_clf_encoder:
            return self.y_clf_encoder.inverse_transform(self.estimator.predict(X))
        else:
            return self.estimator.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)
