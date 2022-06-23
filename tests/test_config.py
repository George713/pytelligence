import pycarrot as pc

config = pc.init_config("./config.yml")


def test_init_config():
    assert type(config) == dict


def test_modelling():
    assert "modelling" in config.keys()


def test_target_clf():
    assert "target_clf" in config["modelling"].keys()


def test_numeric_cols():
    assert "numeric_cols" in config["modelling"].keys()
