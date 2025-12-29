import pytest

from msmu import tl


@pytest.mark.parametrize("attr", tl.__all__)
def test_tools_public_api_has_attributes(attr):
    assert hasattr(tl, attr), f"{attr} missing from tools module"
