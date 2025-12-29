import pytest

import msmu


@pytest.mark.parametrize("name", msmu.__all__)
def test_package_exports_accessible(name):
    assert hasattr(msmu, name), f"{name} missing from package exports"
