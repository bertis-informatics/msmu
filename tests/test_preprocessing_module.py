import pytest

from msmu import pp


@pytest.mark.parametrize("func_name", pp.__all__)
def test_preprocessing_public_api_has_attributes(func_name):
    assert hasattr(pp, func_name), f"{func_name} missing from preprocessing module"
