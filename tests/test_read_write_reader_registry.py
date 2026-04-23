import inspect
import logging
from pathlib import Path

import pandas as pd
from mudata import MuData

from msmu._read_write import _reader_registry as rr


def _dummy_mdata() -> MuData:
    from anndata import AnnData
    import numpy as np

    adata = AnnData(X=np.array([[1.0]]), obs=pd.DataFrame(index=["s1"]), var=pd.DataFrame(index=["f1"]))
    return MuData({"psm": adata})


def _dummy_convert(self, file_paths, max_workers=4):
    return file_paths[0], pd.DataFrame()


class _DummyReader:
    def __init__(self, *args, **kwargs):
        pass

    def read(self):
        logging.getLogger("msmu.test").info("reader log")
        return _dummy_mdata()


def _assert_cmd(mdata: MuData, expected_function: str, expect_stdout: bool = True):
    assert "_cmd" in mdata.uns
    entry = mdata.uns["_cmd"]["0"]
    assert entry["function"] == expected_function
    assert "msmu_version" in entry
    assert "python_version" in entry
    payload = entry["payload"]
    assert isinstance(payload, dict)
    if expect_stdout:
        assert "stdout" in entry
        assert "INFO - reader log" in entry["stdout"]


def test_read_h5mu_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr.md, "read_h5mu", lambda _: _dummy_mdata())
    out = rr.read_h5mu(Path("dummy.h5mu"))
    _assert_cmd(out, "read_h5mu", expect_stdout=False)


def test_public_readers_only_accept_sdrf_file_and_validate_by_default():
    readers = [
        rr.read_sage,
        rr.read_diann,
        rr.read_maxquant,
        rr.read_fragpipe,
        rr.read_delpi,
        rr.read_sdrf,
    ]

    for reader in readers:
        parameters = inspect.signature(reader).parameters
        assert "sdrf_path" not in parameters
        assert "sdrf_file" in parameters
        assert parameters["validate_sdrf"].default is True


def test_read_sage_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr.SearchResultDataFrameConverter, "convert", _dummy_convert)
    monkeypatch.setattr(rr, "TmtSageReader", _DummyReader)
    out = rr.read_sage("id.tsv", label="tmt", quantification_file="quant.tsv")
    _assert_cmd(out, "read_sage")


def test_read_sage_label_free_without_quant_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr.SearchResultDataFrameConverter, "convert", _dummy_convert)
    monkeypatch.setattr(rr, "LfqSageReader", _DummyReader)
    out = rr.read_sage("id.tsv", label="label_free")
    _assert_cmd(out, "read_sage")


def test_read_diann_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr.SearchResultDataFrameConverter, "convert", _dummy_convert)
    monkeypatch.setattr(rr, "DiannReader", _DummyReader)
    out = rr.read_diann("id.tsv")
    _assert_cmd(out, "read_diann")


def test_read_diann_passes_sdrf_file_and_default_validation_to_attach(monkeypatch):
    calls = []

    def attach(mdata, sdrf_file, *, validate):
        calls.append((sdrf_file, validate))
        return mdata

    monkeypatch.setattr(rr.SearchResultDataFrameConverter, "convert", _dummy_convert)
    monkeypatch.setattr(rr, "DiannReader", _DummyReader)
    monkeypatch.setattr(rr, "attach_sdrf_metadata", attach)

    out = rr.read_diann("id.tsv", sdrf_file="meta.sdrf.tsv")

    _assert_cmd(out, "read_diann")
    assert calls == [("meta.sdrf.tsv", True)]


def test_read_diann_protein_group_not_implemented():
    import pytest
    with pytest.raises(NotImplementedError):
        rr.read_diann("id.tsv", level="protein_group")


def test_read_maxquant_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr.MaxQuantDataFrameConverter, "convert", _dummy_convert)
    monkeypatch.setattr(rr, "MaxTmtReader", _DummyReader)
    out = rr.read_maxquant("id.tsv", label="tmt", acquisition="dda")
    _assert_cmd(out, "read_maxquant")


def test_read_fragpipe_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr.SearchResultDataFrameConverter, "convert", _dummy_convert)
    monkeypatch.setattr(rr, "TmtFragPipeReader", _DummyReader)
    out = rr.read_fragpipe("id.tsv", label="tmt", acquisition="dda")
    _assert_cmd(out, "read_fragpipe")
