import pytest

import pytelligence as pt


@pytest.fixture(scope="module")
def config():
    return pt.modelling._prepare_data._init_config(path="./tests/config_test.yml")
