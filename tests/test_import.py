import importlib


def test_basic_import():
    module = importlib.import_module("msmu")
    assert module.__name__ == "msmu"
