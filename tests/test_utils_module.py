import pytest

from msmu import utils


@pytest.mark.parametrize("name", utils.__all__)
def test_utils_public_api_has_attributes(name):
    assert hasattr(utils, name), f"{name} missing from utils module"
