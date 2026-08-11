import pytest

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

import msmu as mm
from msmu._tools import _pyopenms


def test_precursor_isolation_purity_is_tools_api_only():
    assert not hasattr(mm.pp, "compute_precursor_isolation_purity")
    assert mm.tl.compute_precursor_isolation_purity is _pyopenms.compute_precursor_isolation_purity
    assert mm.tl.compute_precursor_isolation_purity_from_mzml is _pyopenms.compute_precursor_isolation_purity_from_mzml


def test_purity_result_plotting_uses_public_plotting_facade(monkeypatch):
    calls = []
    sentinel = object()

    def fake_plot_var(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(mm.pl, "plot_var", fake_plot_var)

    purity = _pyopenms.PurityResult(
        purity=[0.8],
        scan_num=[1],
        filename=["run1.mzML"],
    )

    assert purity.hist() is sentinel
    assert purity.box() is sentinel
    assert [call["ptype"] for call in calls] == ["hist", "box"]
    assert {call["var_column"] for call in calls} == {"purity"}


def test_compute_precursor_isolation_purity_assigns_psm_var_purity(monkeypatch):
    class FakePrecursorPurityCalculator:
        @classmethod
        def from_mudata(cls, mdata, tolerance=20.0, unit_ppm=True):
            assert tolerance == 20.0
            assert unit_ppm is True
            return cls()

        @property
        def mzml(self):
            return self._mzml

        @mzml.setter
        def mzml(self, value):
            self._mzml = value

        def calculate_precursor_isolation_purities(self):
            return pd.DataFrame(
                {
                    "filename": ["run1.mzML", "run1.mzML"],
                    "scan_num": [1, 2],
                    "purity": [0.8, 0.9],
                }
            )

    psm_var = pd.DataFrame(
        {"filename": ["run1", "run1"], "scan_num": [1, 2]},
        index=["run1.1", "run1.2"],
    )
    psm = AnnData(np.ones((1, 2)), obs=pd.DataFrame(index=["sample1"]), var=psm_var)
    mdata = MuData({"psm": psm})

    monkeypatch.setattr(_pyopenms, "PrecursorPurityCalculator", FakePrecursorPurityCalculator)

    out = mm.tl.compute_precursor_isolation_purity(mdata, mzml_paths=["/data/run1.mzML"])

    assert out is not mdata
    assert "purity" not in mdata.mod["psm"].var
    assert out.mod["psm"].var["purity"].tolist() == [0.8, 0.9]


def test_compute_precursor_isolation_purity_validates_mzml_paths_type(mdata):
    with pytest.raises(TypeError, match="mzml_paths must be a string, Path, or list"):
        mm.tl.compute_precursor_isolation_purity(mdata, mzml_paths=object())


def test_precursor_purity_from_mudata_missing_required_columns_raises():
    class CalculatorWithoutOpenMSInit(_pyopenms.PrecursorPurityCalculator):
        def __init__(self, tolerance=20.0, unit_ppm=True):
            self._var_df = None

    psm_var = pd.DataFrame({"filename": ["run1"]}, index=["run1.1"])
    psm = AnnData(np.ones((1, 1)), obs=pd.DataFrame(index=["sample1"]), var=psm_var)
    mdata = MuData({"psm": psm})

    with pytest.raises(ValueError, match=r"Required columns missing from psm.var: \['scan_num'\]"):
        CalculatorWithoutOpenMSInit.from_mudata(mdata)
