import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData
from scipy import sparse as sp

from msmu._core._status import MuDataStatus


def _status_for_matrix(matrix):
    obs = pd.DataFrame(index=[f"s{i}" for i in range(matrix.shape[0])])
    var = pd.DataFrame(index=[f"f{i}" for i in range(matrix.shape[1])])
    return MuDataStatus(MuData({"psm": AnnData(matrix, obs=obs, var=var)}))


def test_mudata_status_has_quant_for_dense_non_nan_matrix():
    status = _status_for_matrix(np.array([[np.nan, 1.0]]))

    assert status.psm is not None
    assert status.psm.has_quant


def test_mudata_status_has_no_quant_without_matrix():
    adata = AnnData(obs=pd.DataFrame(index=["s1"]), var=pd.DataFrame(index=["f1"]))
    status = MuDataStatus(MuData({"psm": adata}))

    assert status.psm is not None
    assert not status.psm.has_quant


def test_mudata_status_has_no_quant_for_dense_all_nan_matrix():
    status = _status_for_matrix(np.array([[np.nan, np.nan]]))

    assert status.psm is not None
    assert not status.psm.has_quant


def test_mudata_status_has_no_quant_for_sparse_only_stored_nan():
    """A sparse matrix whose only stored value is NaN (the rest structurally absent) has no quant.

    Matches _has_quant_values' contract ("at least one non-NaN value"): there is none here. The
    previous "nnz < rows*cols -> True" shortcut reported quant purely from the sparse structure,
    which also mis-reported an all-absent (nnz==0) matrix (§C-4).
    """
    status = _status_for_matrix(sp.csr_matrix(([np.nan], ([0], [0])), shape=(1, 2)))

    assert status.psm is not None
    assert not status.psm.has_quant


def test_mudata_status_has_no_quant_for_sparse_all_nan_matrix():
    status = _status_for_matrix(sp.csr_matrix([[np.nan, np.nan]]))

    assert status.psm is not None
    assert not status.psm.has_quant


@pytest.mark.parametrize("matrix", [np.empty((0, 0)), sp.csr_matrix((0, 0))])
def test_mudata_status_has_no_quant_for_empty_matrix(matrix):
    status = _status_for_matrix(matrix)

    assert status.psm is not None
    assert not status.psm.has_quant
