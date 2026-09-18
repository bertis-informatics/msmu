from msmu._provenance import get_log
import io
import logging
import pytest
from mudata import MuData

from msmu.logging_utils import get_logger
from msmu._utils.peptide import (
    _calc_exp_mz,
    _count_missed_cleavages,
    _get_peptide_length,
    _make_stripped_peptide,
)
import msmu._utils as msmu_utils
from msmu._core._provenance import _serialize_parameters, log_provenance


def test_serialize_nested_objects():
    obj = {"a": (1, 2), "b": {"c": [3, 4]}}
    out = _serialize_parameters(obj)
    assert out == {"a": [1, 2], "b": {"c": [3, 4]}}


def test_log_provenance_adds_log_entry(labeled_mdata):
    @log_provenance
    def dummy(mdata: MuData, value: int):
        return mdata

    out = dummy(labeled_mdata, value=3)
    assert "_log" in out.uns
    entry = get_log(out)["events"][0]
    assert entry["function"] == "dummy"
    assert entry["environment_id"] in get_log(out)["environments"]
    assert entry["inputs"][0]["type"] == "MuData"
    assert entry["outputs"][0]["type"] == "MuData"
    assert set(entry["inputs"][0]) == {"id", "role", "type", "hash"}
    payload = entry["parameters"]
    assert payload["value"] == 3


def test_log_provenance_records_default_parameters(labeled_mdata):
    @log_provenance
    def dummy_defaults(mdata: MuData, value: int = 10, flag: bool = False):
        return mdata

    out = dummy_defaults(labeled_mdata)
    entry = get_log(out)["events"][0]
    payload = entry["parameters"]
    assert payload["value"] == 10
    assert payload["flag"] is False


def test_log_provenance_preserves_stdout_without_storing_it(labeled_mdata, capsys):
    @log_provenance
    def dummy_print(mdata: MuData):
        print("hello from print")
        return mdata

    out = dummy_print(labeled_mdata)
    entry = get_log(out)["events"][0]
    assert "stdout" not in entry
    assert "hello from print" in capsys.readouterr().out


def test_log_provenance_preserves_logging_without_storing_it(labeled_mdata, caplog):
    @log_provenance
    def dummy_log(mdata: MuData):
        logging.getLogger("msmu.test").info("hello from logger")
        return mdata

    with caplog.at_level(logging.INFO, logger="msmu"):
        out = dummy_log(labeled_mdata)
    entry = get_log(out)["events"][0]
    assert "stdout" not in entry
    assert "hello from logger" in caplog.text


def test_log_provenance_prunes_closed_msmu_handler(labeled_mdata):
    logger = get_logger()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        stream = io.StringIO()
        stale_handler = logging.StreamHandler(stream)
        stale_handler._msmu_handler = True  # type: ignore[attr-defined]
        logger.handlers = [stale_handler]
        stream.close()

        @log_provenance
        def dummy_log(mdata: MuData):
            logging.getLogger("msmu.test").info("hello after closed handler")
            return mdata

        out = dummy_log(labeled_mdata)

        assert stale_handler not in logger.handlers
        entry = get_log(out)["events"][0]
        assert "stdout" not in entry
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_log_provenance_keeps_existing_events_intact_across_multiple_calls(
    labeled_mdata,
):
    @log_provenance
    def dummy(mdata: MuData):
        return mdata

    mdata = labeled_mdata.copy()
    mdata.mod["psm"].layers["raw"] = mdata.mod["psm"].X.copy()

    out = dummy(mdata)
    original_entry = get_log(out)["events"][0]
    out = dummy(out)

    first_entry = get_log(out)["events"][0]
    second_entry = get_log(out)["events"][1]

    assert first_entry == original_entry
    assert second_entry["parents"] == [first_entry["id"]]


def test_provenance_rejects_invalid_log_before_execution(labeled_mdata):
    labeled_mdata.uns["_log"] = "invalid"
    with pytest.raises(ValueError, match="Unsupported"):
        log_provenance(lambda mdata: mdata)(labeled_mdata)


def test_utils_all_exports_are_bound():
    for name in msmu_utils.__all__:
        assert hasattr(msmu_utils, name)


def test_peptide_helpers():
    assert _make_stripped_peptide("ACD[+57.02]EF") == "ACDEF"
    assert _count_missed_cleavages("AKRP") == 1
    assert _get_peptide_length("ACD") == 3
    assert _calc_exp_mz(100.0, 2) == pytest.approx(51.007276466812)
