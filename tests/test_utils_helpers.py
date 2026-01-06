import numpy as np
import pandas as pd
import pytest
from mudata import MuData

from msmu._utils.peptide import (
    _calc_exp_mz,
    _count_missed_cleavages,
    _get_peptide_length,
    _make_stripped_peptide,
)
from msmu._utils.utils import add_quant, get_label, get_modality_dict, serialize, uns_logger


def test_serialize_nested_objects():
    obj = {"a": (1, 2), "b": {"c": [3, 4]}}
    out = serialize(obj)
    assert out == {"a": [1, 2], "b": {"c": [3, 4]}}


def test_uns_logger_adds_cmd_entry(labeled_mdata):
    @uns_logger
    def dummy(mdata: MuData, value: int):
        return mdata

    out = dummy(labeled_mdata, value=3)
    assert "_cmd" in out.uns
    assert out.uns["_cmd"]["0"]["function"] == "dummy"
    assert out.uns["_cmd"]["0"]["kwargs"]["value"] == "3"


def test_get_modality_dict_by_modality(labeled_mdata):
    mdata = labeled_mdata
    mods = get_modality_dict(mdata, modality="psm")
    assert "psm" in mods


def test_get_label_from_psm(labeled_mdata):
    mdata = labeled_mdata
    assert get_label(mdata) == "tmt"


def test_add_quant_flashlfq_adds_modality(labeled_mdata):
    mdata = labeled_mdata
    quant = pd.DataFrame(
        {
            "Sequence": ["AA", "BB"],
            "Intensity_s1": [1.0, 0.0],
            "Intensity_s2": [2.0, 3.0],
            "Intensity_gis1": [0.0, 4.0],
        }
    )
    out = add_quant(mdata, quant_data=quant, quant_tool="flashlfq")
    assert "peptide" in out.mod_names
    assert out["peptide"].uns["level"] == "peptide"


def test_peptide_helpers():
    assert _make_stripped_peptide("ACD[+57.02]EF") == "ACDEF"
    assert _count_missed_cleavages("AKRP") == 1
    assert _get_peptide_length("ACD") == 3
    assert _calc_exp_mz(100.0, 2) == pytest.approx(51.007276466812)
