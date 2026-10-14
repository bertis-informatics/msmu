import anndata as ad
import mudata as md
import numpy as np
import pytest
from scipy import sparse

import msmu as mm


@pytest.mark.parametrize("sparse_input", [False, True])
def test_save_and_load_layer_copy_and_record(sparse_input):
    original = np.array([[1.0, 0.0], [0.0, 2.0]])
    x = sparse.csr_matrix(original) if sparse_input else original.copy()
    mdata = md.MuData({"protein": ad.AnnData(X=x)})
    adata = mdata["protein"]

    with mm.pv.options(hashing=False):
        assert mm.dt.save_layer(mdata, modality="protein", layer="raw") is mdata
        layer = adata.layers["raw"]
        assert sparse.issparse(layer) == sparse_input

        if sparse_input:
            adata.X.data[0] = 9.0
        else:
            adata.X[0, 0] = 9.0
        np.testing.assert_array_equal(layer.toarray() if sparse_input else layer, original)

        with pytest.raises(ValueError):
            mm.dt.save_layer(mdata, modality="protein", layer="raw")
        with pytest.raises(KeyError):
            mm.dt.load_layer(mdata, modality="protein", layer="missing")

        assert mm.dt.load_layer(mdata, modality="protein", layer="raw") is mdata
        assert sparse.issparse(adata.X) == sparse_input
        np.testing.assert_array_equal(adata.X.toarray() if sparse_input else adata.X, original)

        if sparse_input:
            adata.X.data[0] = 7.0
        else:
            adata.X[0, 0] = 7.0
        np.testing.assert_array_equal(layer.toarray() if sparse_input else layer, original)

        assert mm.dt.save_layer(mdata, modality="protein", layer="raw", overwrite=True) is mdata
        expected = original.copy()
        expected[0, 0] = 7.0
        np.testing.assert_array_equal(
            adata.layers["raw"].toarray() if sparse_input else adata.layers["raw"], expected
        )

    assert [event["function"] for event in mm.pv.get_log(mdata)["events"]] == [
        "save_layer", "load_layer", "save_layer"
    ]


def test_layer_roundtrip_replays_from_h5mu(tmp_path):
    source = tmp_path / "source.h5mu"
    md.MuData({"protein": ad.AnnData(X=np.array([[2.0, 4.0], [8.0, 16.0]]))}).write_h5mu(source)

    with mm.pv.options(hashing=True):
        mdata = mm.read_h5mu(source)
        mdata = mm.dt.save_layer(mdata, modality="protein", layer="raw")
        mdata = mm.pp.log2_transform(mdata, modality="protein")
        mdata = mm.dt.load_layer(mdata, modality="protein", layer="raw")

    assert mm.pv.compute_hash(mm.pv.replay(mdata, verify=True)) == mm.pv.compute_hash(mdata)


@pytest.mark.parametrize("mode", ["r", "r+"])
def test_layer_operations_reject_backed_mudata(tmp_path, mode):
    source = tmp_path / "source.h5mu"
    original = np.array([[2.0, 4.0]])
    md.MuData({"protein": ad.AnnData(X=original)}).write_h5mu(source)
    mdata = md.read_h5mu(source, backed=mode)
    try:
        layers_before = list(mdata["protein"].layers)
        with pytest.raises(ValueError, match="backed"):
            mm.dt.save_layer(mdata, modality="protein", layer="raw")
        with pytest.raises(ValueError, match="backed"):
            mm.dt.load_layer(mdata, modality="protein", layer="raw")
        assert list(mdata["protein"].layers) == layers_before
        np.testing.assert_array_equal(mdata["protein"].X[:], original)
        assert mm.pv.get_log(mdata)["events"] == []
    finally:
        mdata.file.close()
