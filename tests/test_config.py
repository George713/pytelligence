def test_init_config(config):
    assert type(config) == dict


def test_key_modelling(config):
    assert "modelling" in config.keys()


def test_modelling(config):
    assert all(
        key in config["modelling"].keys()
        for key in [
            "target_clf",
            "numeric_cols",
            "categorical_cols",
            "feature_scaling",
        ]
    )
