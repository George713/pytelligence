import pytelligence as pt

config = pt.modelling._prepare_data._init_config(path="./tests/config_test.yml")


def test_init_config():
    assert type(config) == dict


def test_key_modelling():
    assert "modelling" in config.keys()


def test_modelling():
    assert all(
        key in config["modelling"].keys()
        for key in [
            "target_clf",
            "numeric_cols",
            "categorical_cols",
            "feature_scaling",
        ]
    )
