from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msmu import io as msmu_io
import msmu._utils as msmu_utils


def test_add_quant_flashlfq_file_adds_peptide_modality(labeled_mdata):
    quant_file = Path(__file__).with_name("flashlfq_quant.tsv")

    out = msmu_io.add_quant(labeled_mdata, quant_data=quant_file, quant_tool="flashlfq")

    assert "peptide" in out.mod
    assert out["peptide"].uns["level"] == "peptide"
    assert list(out["peptide"].obs_names) == ["s1", "s2", "gis1"]
    assert list(out["peptide"].var_names) == ["AA", "BB"]
    assert np.isnan(out["peptide"].to_df().loc["gis1", "AA"])
    assert out["peptide"].to_df().loc["gis1", "BB"] == pytest.approx(4.0)


def test_add_quant_is_not_exported_from_utils():
    assert "add_quant" not in msmu_utils.__all__
    assert not hasattr(msmu_utils, "add_quant")


def test_add_quant_rejects_unknown_quant_tool(labeled_mdata):
    with pytest.raises(ValueError, match="Unsupported quant_tool"):
        msmu_io.add_quant(labeled_mdata, quant_data=pd.DataFrame(), quant_tool="unknown")


def test_add_quant_flashlfq_requires_sequence_column(labeled_mdata):
    quant = pd.DataFrame({"Intensity_s1": [1.0]})

    with pytest.raises(ValueError, match="Sequence"):
        msmu_io.add_quant(labeled_mdata, quant_data=quant, quant_tool="flashlfq")
