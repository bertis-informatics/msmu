"""A wide `precursor` modality gives the quantification and the scores the long `psm` modality gives.

The same identifications are laid out twice: long (one variable per run and precursor, as the readers
build it) and wide (observations = runs, variables = precursors, scores in layers, as an external
file brings it). Filtering on the identification score and rolling up must not depend on the layout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu as mm

DIANN_REPORT = Path(__file__).resolve().parents[1] / "data" / "diann_dia" / "diann" / "report.parquet"
PRECURSOR_COMMON_COLUMNS = [
    "proteins",
    "peptide",
    "stripped_peptide",
    "charge",
    "peptide_length",
    "contaminant",
    "decoy",
]
SCORE_COLUMNS = ["q_value", "PEP"]
FLOAT32_RELATIVE_TOLERANCE = 1e-6


def _get_precursor_names(long_var: pd.DataFrame) -> pd.Index:
    """Precursor names of long variables, which are named "<run>.<precursor>"."""
    run_names = long_var["filename"].astype(str)
    return pd.Index([variable_name[len(run_name) + 1 :] for variable_name, run_name in zip(long_var.index, run_names)])


def _convert_long_psm_to_wide_precursor(long_mdata: MuData) -> MuData:
    """Lay a long `psm` modality out as the wide `precursor` modality an external file would bring."""
    long_psm = long_mdata["psm"]
    long_var = long_psm.var
    run_names = long_var["filename"].astype(str)
    precursor_names = _get_precursor_names(long_var)
    wide_precursor_names = precursor_names.unique()
    run_positions = long_psm.obs_names.get_indexer(run_names)
    precursor_positions = wide_precursor_names.get_indexer(precursor_names)
    assert (run_positions >= 0).all()

    def spread_over_runs(values_per_identification: np.ndarray) -> np.ndarray:
        wide_matrix = np.full((long_psm.n_obs, len(wide_precursor_names)), np.nan, dtype="float32")
        wide_matrix[run_positions, precursor_positions] = values_per_identification
        return wide_matrix

    # A long variable stores its one value in its own run; a variable without a stored value is missing.
    long_quantification = long_psm.X.tocsc()
    assert long_quantification.getnnz(axis=0).max() <= 1
    quantification_per_identification = np.asarray(
        long_quantification[run_positions, np.arange(long_psm.n_vars)]
    ).ravel()
    quantification_per_identification[long_quantification.getnnz(axis=0) == 0] = np.nan

    common_values_by_precursor = long_var[PRECURSOR_COMMON_COLUMNS].groupby(
        precursor_names.to_numpy(), sort=False, observed=True
    )
    assert (common_values_by_precursor.nunique() == 1).all().all()

    wide_precursor = AnnData(
        X=spread_over_runs(quantification_per_identification),
        obs=long_psm.obs.copy(),
        var=common_values_by_precursor.first().loc[wide_precursor_names],
    )
    for score_column in SCORE_COLUMNS:
        wide_precursor.layers[score_column] = spread_over_runs(long_var[score_column].to_numpy(dtype="float32"))
    wide_precursor.uns.update(long_psm.uns)

    return MuData({"precursor": wide_precursor})


def _roll_up_to_protein(mdata: MuData, source_modality: str) -> MuData:
    mdata = mm.pp.log2_transform(mdata, modality=source_modality)
    mdata = mm.pp.normalise(mdata, method="median", modality=source_modality)
    mdata = mm.pp.to_peptide(mdata)
    mdata = mm.pp.infer_protein(mdata)
    return mm.pp.to_protein(mdata)


def _assert_same_quantification(wide_modality: AnnData, long_modality: AnnData) -> None:
    assert wide_modality.obs_names.tolist() == long_modality.obs_names.tolist()
    assert wide_modality.var_names.tolist() == long_modality.var_names.tolist()
    wide_values = np.asarray(wide_modality.X, dtype=float)
    long_values = np.asarray(long_modality.X, dtype=float)
    np.testing.assert_array_equal(np.isnan(wide_values), np.isnan(long_values))
    # The long psm matrix is float32 (sparse) through log2 and normalisation, the wide one is computed
    # in float64, so the two agree to float32 precision rather than bit for bit.
    np.testing.assert_allclose(wide_values, long_values, rtol=FLOAT32_RELATIVE_TOLERANCE, equal_nan=True)


@pytest.mark.parametrize("score_column, threshold", [("q_value", 0.005), ("PEP", 0.01)])
def test_wide_precursor_rolls_up_to_the_long_psm_quantification_and_scores(score_column, threshold):
    long_mdata = mm.read_diann(identification_file=str(DIANN_REPORT))
    wide_mdata = _convert_long_psm_to_wide_precursor(long_mdata)
    assert wide_mdata["precursor"].n_vars < long_mdata["psm"].n_vars

    long_filtered = mm.pp.add_filter(long_mdata, modality="psm", column=score_column, keep="lt", value=threshold)
    long_filtered = mm.pp.apply_filter(long_filtered, modality="psm")
    wide_filtered = mm.pp.filter(
        wide_mdata, modality="precursor", on="layers", column=score_column, keep="lt", value=threshold
    )

    # The threshold removes identifications, and the same ones in both layouts.
    assert long_filtered["psm"].n_vars < long_mdata["psm"].n_vars
    assert int(np.isfinite(wide_filtered["precursor"].layers[score_column]).sum()) == long_filtered["psm"].n_vars
    # The decoy precursors are filtered with the targets, on their own cells, and some fail.
    wide_decoy_names = wide_filtered["precursor"].var.query("decoy == 1").index
    long_decoy_names = _get_precursor_names(long_filtered["psm"].var.query("decoy == 1")).unique()
    assert 0 < len(wide_decoy_names) < (wide_mdata["precursor"].var["decoy"] == 1).sum()
    assert sorted(wide_decoy_names) == sorted(long_decoy_names)

    long_result = _roll_up_to_protein(long_filtered, source_modality="psm")
    wide_result = _roll_up_to_protein(wide_filtered, source_modality="precursor")

    _assert_same_quantification(wide_result["peptide"], long_result["peptide"])
    _assert_same_quantification(wide_result["protein"], long_result["protein"])
    for annotation_column in ("proteins", "stripped_peptide", "protein_group", "decoy"):
        assert (
            wide_result["peptide"].var[annotation_column].astype(str).tolist()
            == long_result["peptide"].var[annotation_column].astype(str).tolist()
        )
    # The best PEP of the layer is the best PEP of the long rows, so the q-values of the targets and
    # decoys follow.
    for level in ("peptide", "protein"):
        assert (wide_result[level].var["decoy"] == 1).any()
        for score_column in SCORE_COLUMNS:
            np.testing.assert_array_equal(
                wide_result[level].var[score_column].to_numpy(), long_result[level].var[score_column].to_numpy()
            )
        # A wide variable is a precursor, not an identification, so identifications are not counted.
        assert "count_psm" in long_result[level].var.columns
        assert "count_psm" not in wide_result[level].var.columns
