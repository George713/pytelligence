from . import feat_analysis
from . import modelling

import yaml


def init_config(path: str):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
