from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from msmu import io as msmu_io
import msmu._utils as msmu_utils


def _identification_only_peptide_mdata() -> md.MuData:
    """A peptide modality carrying identifications but no quant (the state to_peptide leaves
    behind when it runs before add_quant on label-free data). ``CC`` is identified but will
    not be quantified by FlashLFQ."""
    peptide = ad.AnnData(
        X=np.full((3, 3), np.nan, dtype=np.float64),
        obs=pd.DataFrame(index=["s1", "s2", "gis1"]),
        var=pd.DataFrame(
            {"proteins": ["P1", "P2", "P3"], "q_value": [0.001, 0.002, 0.02]},
            index=["AA", "BB", "CC"],
        ),
    )
    peptide.uns["level"] = "peptide"
    peptide.uns["decoy"] = pd.DataFrame({"score": [9.0]}, index=["DECOY1"])
    return md.MuData({"peptide": peptide})


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


def test_add_quant_fills_existing_peptide_modality_and_preserves_identification():
    # Regression (BID-134): when a peptide identification layer already exists (to_peptide ran
    # before add_quant), add_quant must NOT overwrite it. It fills quant aligned onto the
    # existing peptide/sample axes while keeping var (proteins, q_value) and uns (decoy).
    mdata = _identification_only_peptide_mdata()
    quant = pd.DataFrame(
        {
            "Sequence": ["AA", "BB", "ZZ"],  # ZZ is quantified but not identified -> dropped
            "Intensity_s1": [1000.0, 3000.0, 7000.0],
            "Intensity_s2": [2000.0, 5000.0, 8000.0],
            "Intensity_gis1": [0.0, 4.0, 9.0],  # 0 -> NaN
        }
    )

    out = msmu_io.add_quant(mdata, quant_data=quant, quant_tool="flashlfq")
    peptide = out["peptide"]

    # identification metadata survives untouched
    assert list(peptide.var.columns) == ["proteins", "q_value"]
    assert list(peptide.var["proteins"]) == ["P1", "P2", "P3"]
    assert peptide.uns["level"] == "peptide"
    assert "decoy" in peptide.uns

    # peptide axis is the identification set: CC kept (identified), ZZ dropped (not identified)
    assert list(peptide.var_names) == ["AA", "BB", "CC"]

    # quant is ingested as float32 (msmu .X convention) regardless of the container's prior dtype
    assert peptide.X.dtype == np.float32

    # quant filled by sequence+sample; identified-but-unquantified stays NaN
    df = peptide.to_df()
    assert df.loc["s1", "AA"] == pytest.approx(1000.0)
    assert df.loc["s2", "BB"] == pytest.approx(5000.0)
    assert df.loc["gis1", "BB"] == pytest.approx(4.0)
    assert np.isnan(df.loc["gis1", "AA"])  # 0 intensity -> NaN
    assert bool(np.isnan(df.loc[:, "CC"]).all())  # CC identified but not quantified


def test_add_quant_does_not_wipe_identification_var_regression():
    # The original bug produced an empty var (add_modality overwrote the whole modality).
    mdata = _identification_only_peptide_mdata()
    quant = pd.DataFrame(
        {"Sequence": ["AA"], "Intensity_s1": [1.0], "Intensity_s2": [2.0], "Intensity_gis1": [3.0]}
    )

    out = msmu_io.add_quant(mdata, quant_data=quant, quant_tool="flashlfq")

    assert out["peptide"].var.shape[1] > 0
    assert "q_value" in out["peptide"].var.columns
