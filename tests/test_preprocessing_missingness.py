"""``calculate_missingness`` records per-feature missing counts in ``var``, for targets and decoys alike.

The sparse fixture stores one explicit NaN besides the structurally absent cells, so a count that
only looked at ``nnz`` (or only at stored NaN) would disagree with the dense count.
"""

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import msmu as mm

SAMPLES = ["s1", "s2", "s3", "s4"]
FEATURES = ["v1", "v2", "v3", "v4"]

# v1 complete, v2 missing in s2, v3 missing in the whole "B" group, v4 missing everywhere.
_VALUES = np.array(
    [
        [1.0, 2.0, 3.0, np.nan],  # s1 (A)
        [1.0, np.nan, 3.0, np.nan],  # s2 (A)
        [1.0, 2.0, np.nan, np.nan],  # s3 (B)
        [1.0, 2.0, np.nan, np.nan],  # s4 (B)
    ]
)
EXPECTED_N_MISSING = [0, 1, 2, 4]
EXPECTED_RATIO_A = [0.0, 0.5, 0.0, 1.0]
EXPECTED_RATIO_B = [0.0, 0.0, 1.0, 1.0]

DECOY_FEATURES = ["d1", "d2"]
# d1 has a value in one sample of each group, d2 in none.
_DECOY_VALUES = np.array(
    [
        [5.0, np.nan],  # s1 (A)
        [np.nan, np.nan],  # s2 (A)
        [5.0, np.nan],  # s3 (B)
        [np.nan, np.nan],  # s4 (B)
    ]
)


def _sparse_with_absent_cells(dense: np.ndarray) -> sp.csr_matrix:
    rows, cols = np.nonzero(~np.isnan(dense))
    matrix = sp.csr_matrix((dense[rows, cols], (rows, cols)), shape=dense.shape)
    # Store one NaN explicitly (s1, v4): absent and stored-NaN must both count as missing.
    matrix = matrix.tolil()
    matrix[0, 3] = np.nan
    return matrix.tocsr()


def _mdata(x_array, *, decoy_values: np.ndarray | None = None) -> md.MuData:
    """The FEATURES, followed by the DECOY_FEATURES (`decoy == 1`) holding ``decoy_values`` when given."""
    obs = pd.DataFrame(
        {"group": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B", "unused"])},
        index=SAMPLES,
    )
    var = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=FEATURES)
    if decoy_values is not None:
        x_array = np.hstack([x_array, decoy_values])
        decoy_var = pd.DataFrame({"score": [0.5, 0.6], "decoy": 1}, index=DECOY_FEATURES)
        var = pd.concat([var.assign(decoy=0), decoy_var])
    adata = ad.AnnData(X=x_array, obs=obs, var=var)
    return md.MuData({"protein": adata})


@pytest.mark.parametrize("make_x", [lambda: _VALUES.copy(), lambda: _sparse_with_absent_cells(_VALUES)])
def test_counts_match_between_dense_and_sparse_and_cover_groups(make_x):
    original = _mdata(make_x())
    result = mm.pp.calculate_missingness(original, modality="protein", group_obs="group")

    var = result["protein"].var
    assert var["n_missing"].tolist() == EXPECTED_N_MISSING
    assert var["missing_ratio"].tolist() == [n / 4 for n in EXPECTED_N_MISSING]
    assert var["missing_ratio_A"].tolist() == EXPECTED_RATIO_A
    assert var["missing_ratio_B"].tolist() == EXPECTED_RATIO_B
    assert "missing_ratio_unused" not in var  # category without samples is skipped
    assert var["score"].tolist() == [1.0, 2.0, 3.0, 4.0]  # existing columns untouched

    # The input is left alone: pp functions return a copy.
    assert "n_missing" not in original["protein"].var


def test_decoys_are_counted_from_their_own_quantification(caplog):
    with caplog.at_level("WARNING", logger="msmu"):
        result = mm.pp.calculate_missingness(
            _mdata(_VALUES.copy(), decoy_values=_DECOY_VALUES), modality="protein", group_obs="group"
        )

    decoy_var = result["protein"].var.loc[DECOY_FEATURES]
    assert decoy_var["n_missing"].tolist() == [2, 4]
    assert decoy_var["missing_ratio"].tolist() == [0.5, 1.0]
    assert decoy_var["missing_ratio_A"].tolist() == [0.5, 1.0]
    assert decoy_var["missing_ratio_B"].tolist() == [0.5, 1.0]
    assert result["protein"].var.loc[FEATURES, "n_missing"].tolist() == EXPECTED_N_MISSING
    assert "no decoy has a value" not in caplog.text


def test_filter_on_missingness_removes_a_decoy_without_values_like_a_target():
    mdata = mm.pp.calculate_missingness(_mdata(_VALUES.copy(), decoy_values=_DECOY_VALUES), modality="protein")
    mdata = mm.pp.add_filter(mdata, modality="protein", column="missing_ratio", keep="lt", value=1.0)
    filtered = mm.pp.apply_filter(mdata, modality="protein")
    assert filtered["protein"].var_names.tolist() == ["v1", "v2", "v3", "d1"]


def test_warns_when_no_decoy_has_a_value_in_any_sample(caplog):
    # Sage label-free, for one, does not quantify decoys: a missingness filter would remove them all.
    unquantified_decoy_values = np.full_like(_DECOY_VALUES, np.nan)
    with caplog.at_level("WARNING", logger="msmu"):
        mm.pp.calculate_missingness(_mdata(_VALUES.copy(), decoy_values=unquantified_decoy_values), modality="protein")
    assert "protein: no decoy has a value in any sample" in caplog.text


def test_recomputing_after_removing_samples_overwrites_the_columns():
    mdata = mm.pp.calculate_missingness(_mdata(_VALUES.copy()), modality="protein")
    assert mdata["protein"].var["n_missing"].tolist() == EXPECTED_N_MISSING
    subset = mm.pp.calculate_missingness(mdata[["s1", "s2"], :].copy(), modality="protein")
    assert subset["protein"].var["n_missing"].tolist() == [0, 1, 0, 2]
    assert subset["protein"].var["missing_ratio"].tolist() == [0.0, 0.5, 0.0, 1.0]


def test_missing_group_label_counts_only_towards_overall_columns(caplog):
    mdata = _mdata(_VALUES.copy())
    mdata["protein"].obs["group"] = pd.Series(["A", "A", "B", None], index=SAMPLES, dtype="object")
    with caplog.at_level("WARNING", logger="msmu"):
        result = mm.pp.calculate_missingness(mdata, modality="protein", group_obs="group")
    assert "1 missing labels" in caplog.text
    var = result["protein"].var
    assert var["n_missing"].tolist() == EXPECTED_N_MISSING
    assert var["missing_ratio_B"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert [column for column in var.columns if column.startswith("missing_ratio_")] == [
        "missing_ratio_A", "missing_ratio_B",
    ]


@pytest.mark.parametrize("kwargs, error", [
    ({"group_obs": "absent"}, KeyError),
    ({"group_obs": "with_slash"}, ValueError),
])
def test_invalid_group_obs_raises_and_leaves_input_unchanged(kwargs, error):
    mdata = _mdata(_VALUES.copy())
    mdata["protein"].obs["with_slash"] = ["a/b", "a/b", "c", "c"]
    before = mm.pv.compute_hash(mdata)
    with pytest.raises(error):
        mm.pp.calculate_missingness(mdata, modality="protein", **kwargs)
    assert mm.pv.compute_hash(mdata) == before


def test_no_observations_raises():
    empty = md.MuData({"protein": ad.AnnData(
        X=np.empty((0, len(FEATURES))), obs=pd.DataFrame(index=[]), var=pd.DataFrame(index=FEATURES),
    )})
    with pytest.raises(ValueError, match="no observations"):
        mm.pp.calculate_missingness(empty, modality="protein")


def test_provenance_records_parameters_and_replays(tmp_path):
    source = tmp_path / "source.h5mu"
    _mdata(_VALUES.copy()).write_h5mu(source)
    with mm.pv.options(hashing=True):
        mdata = mm.pv.log(mm.read_h5mu)(source)
        result = mm.pp.calculate_missingness(mdata, modality="protein", group_obs="group")
    expected = result["protein"].var.copy()
    saved = tmp_path / "result.h5mu"
    result.write_h5mu(saved)
    history = mm.pv.get_log(md.read_h5mu(saved))
    event = history["events"][-1]
    assert event["function"] == "calculate_missingness"
    assert {k: v for k, v in event["parameters"].items() if k != "mdata"} == {
        "modality": "protein", "group_obs": "group",
    }
    pd.testing.assert_frame_equal(mm.pv.replay(history)["protein"].var, expected)
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    pd.testing.assert_frame_equal(namespace["mdata"]["protein"].var, expected)
