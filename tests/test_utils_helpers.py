import logging
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
from msmu._utils import append_cmd_log, add_quant, get_label, get_modality_dict, uns_logger
from msmu._utils._provenance import serialize


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
    entry = out.uns["_cmd"]["0"]
    assert entry["function"] == "dummy"
    assert "msmu_version" in entry
    assert "python_version" in entry
    assert "input_dimensions" in entry
    assert "output_dimensions" in entry
    assert entry["input_dimensions"]["n_obs"] == out.n_obs
    assert entry["output_dimensions"]["n_obs"] == out.n_obs
    first_mod = out.mod_names[0]
    assert "layers" in entry["input_dimensions"]["modalities"][first_mod]
    assert isinstance(entry["input_dimensions"]["modalities"][first_mod]["layers"], list)
    payload = entry["payload"]
    assert payload["value"] == 3


def test_uns_logger_records_default_parameters(labeled_mdata):
    @uns_logger
    def dummy_defaults(mdata: MuData, value: int = 10, flag: bool = False):
        return mdata

    out = dummy_defaults(labeled_mdata)
    entry = out.uns["_cmd"]["0"]
    payload = entry["payload"]
    assert payload["value"] == 10
    assert payload["flag"] is False


def test_uns_logger_captures_stdout(labeled_mdata):
    @uns_logger
    def dummy_print(mdata: MuData):
        print("hello from print")
        return mdata

    out = dummy_print(labeled_mdata)
    entry = out.uns["_cmd"]["0"]
    assert "stdout" in entry
    assert "hello from print" in entry["stdout"]


def test_uns_logger_captures_logging(labeled_mdata):
    @uns_logger
    def dummy_log(mdata: MuData):
        logging.getLogger("msmu.test").info("hello from logger")
        return mdata

    out = dummy_log(labeled_mdata)
    entry = out.uns["_cmd"]["0"]
    assert "stdout" in entry
    assert "INFO - hello from logger" in entry["stdout"]


def test_append_cmd_log_rejects_invalid_cmd_type(labeled_mdata):
    mdata = labeled_mdata.copy()
    mdata.uns["_cmd"] = "invalid"
    with pytest.raises(TypeError, match="must be dict or list"):
        append_cmd_log(mdata, function="new")


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
