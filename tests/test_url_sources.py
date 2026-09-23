from msmu._core._provenance import _event_inputs
from io import BytesIO
from urllib.error import URLError

import anndata as ad
import mudata as md
import pandas as pd
import pytest

import msmu as mm
from msmu._core import _sources
from msmu._read_write._base_reader import SearchResultDataFrameConverter
from msmu._provenance import compute_hash, get_log


@pytest.mark.parametrize("suffix", ["csv", "tsv", "parquet"])
@pytest.mark.parametrize("hashing", [False, True])
def test_url_buffer_shared_by_polars_pandas_and_hash(monkeypatch, tmp_path, suffix, hashing):
    expected = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    if suffix == "parquet":
        payload = expected.to_parquet(index=False)
    else:
        payload = expected.to_csv(index=False, sep="," if suffix == "csv" else "\t").encode()
    path = tmp_path / f"input.{suffix}"
    path.write_bytes(payload)
    url = f"https://example.test/input.{suffix}?download=1"
    requests = []
    buffers = []

    def download(source, timeout):
        requests.append(source)
        assert timeout > 0
        return BytesIO(payload)

    monkeypatch.setattr(_sources, "urlopen", download)

    @mm.pv.log
    def load(input_file):
        source, frame = SearchResultDataFrameConverter._read_file(input_file)
        assert source == url
        pd.testing.assert_frame_equal(frame.to_pandas(), expected)
        with _sources.open_source(input_file) as buffer:
            buffers.append(buffer)
            other = pd.read_parquet(buffer) if suffix == "parquet" else pd.read_csv(
                buffer, sep="," if suffix == "csv" else "\t"
            )
        pd.testing.assert_frame_equal(other, expected)
        return md.MuData({"data": ad.AnnData(frame.to_numpy())})

    with mm.pv.options(hashing=hashing):
        result = load(url)
    event = get_log(result)["events"][0]
    assert "inputs" not in event
    entity = _event_inputs(event)[0]
    assert requests == [url]
    assert buffers[0].closed
    assert _sources.get_download_buffer(url) is None
    assert event["parameters"]["input_file"]["path"] == entity["path"] == url
    if hashing:
        assert entity["hash"]["status"] == "completed"
        assert entity["hash"]["value"] == compute_hash(path)
        assert event["outputs"][0]["hash"]["value"] == compute_hash(result)
    else:
        assert entity["hash"] == {"status": "disabled"}


def test_failed_url_read_releases_scope_without_logging(monkeypatch):
    url = "https://example.test/input.csv"
    mdata = md.MuData({"data": ad.AnnData(pd.DataFrame({"x": [1]}))})
    error = URLError("download failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_sources, "urlopen", fail)

    @mm.pv.log
    def load(mdata, input_file):
        SearchResultDataFrameConverter._read_file(input_file)
        return mdata

    with mm.pv.options(hashing=True), pytest.raises(URLError) as caught:
        load(mdata, url)
    assert caught.value is error
    assert "_log" not in mdata.uns
    assert _sources.get_download_buffer(url) is None
    monkeypatch.setattr(_sources, "urlopen", lambda *args, **kwargs: BytesIO(b"x\n1\n"))
    assert len(get_log(load(mdata, url))["events"]) == 1


def test_buffer_hashing_preserves_position(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"source data")
    with BytesIO(path.read_bytes()) as buffer:
        buffer.seek(3)
        assert compute_hash(buffer) == compute_hash(path)
        assert buffer.tell() == 3
