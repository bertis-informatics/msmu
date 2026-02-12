import logging
from pathlib import Path

from mudata import MuData

from msmu._read_write import _reader_registry as rr


def _dummy_mdata() -> MuData:
    from anndata import AnnData
    import numpy as np
    import pandas as pd

    adata = AnnData(X=np.array([[1.0]]), obs=pd.DataFrame(index=["s1"]), var=pd.DataFrame(index=["f1"]))
    return MuData({"psm": adata})


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


def test_read_sage_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr, "TmtSageReader", _DummyReader)
    out = rr.read_sage("id.tsv", label="tmt", quantification_file="quant.tsv")
    _assert_cmd(out, "read_sage")


def test_read_diann_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr, "DiannReader", _DummyReader)
    out = rr.read_diann("id.tsv")
    _assert_cmd(out, "read_diann")


def test_read_diann_from_pg_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr, "DiannProteinGroupReader", _DummyReader)
    out = rr.read_diann.from_pg("id.tsv")
    _assert_cmd(out, "read_diann.from_pg")


def test_read_maxquant_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr, "MaxTmtReader", _DummyReader)
    out = rr.read_maxquant("id.tsv", label="tmt", acquisition="dda")
    _assert_cmd(out, "read_maxquant")


def test_read_fragpipe_logs_first_command(monkeypatch):
    monkeypatch.setattr(rr, "TmtFragPipeReader", _DummyReader)
    out = rr.read_fragpipe("id.tsv", label="tmt", acquisition="dda", quantification_file="quant.tsv")
    _assert_cmd(out, "read_fragpipe")
