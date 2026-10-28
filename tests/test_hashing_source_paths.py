import pandas as pd
import pytest

import msmu as mm

from test_read_write_sage_parity import _clean_labels, _clean_scan_tokens, _write_pair


@pytest.mark.parametrize("drop_search_result", [False, True])
def test_search_reader_sources_live_only_in_provenance_and_relocate(tmp_path, drop_search_result):
    identification, quantification = _write_pair(tmp_path, _clean_scan_tokens(), _clean_labels())
    with pd.option_context("future.infer_string", False), mm.pv.options(hashing=True):
        original = mm.read_sage(identification, label="tmt", quantification_file=quantification,
                                drop_search_result=drop_search_result)
        original = mm.pp.log2_transform(original, modality="psm")
        digest = mm.pv.compute_hash(original)
        reader_event = mm.pv.get_log(original)["events"][0]
        assert reader_event["outputs"][0]["hash"]["normalization"] == "msmu-v2"
        assert reader_event["parameters"]["identification_file"]["path"] == str(identification)
        assert reader_event["parameters"]["quantification_file"]["path"] == str(quantification)

        replacements = {}
        for source in (identification, quantification):
            moved = tmp_path / f"moved-{source.name}"
            source.rename(moved)
            replacements[str(source)] = moved

        replayed = mm.pv.replay(original, sources=replacements)
        namespace = {}
        exec(mm.pv.to_script(original, sources=replacements), namespace)

    for result in (original, replayed, namespace["mdata"]):
        assert mm.pv.compute_hash(result) == digest
        for owner in (result, *result.mod.values()):
            assert "identification_file" not in owner.uns
            assert "quantification_file" not in owner.uns

    with replacements[str(quantification)].open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch at read_sage"):
        mm.pv.replay(original, sources=replacements)
    with pytest.raises(ValueError, match="hash mismatch at read_sage"):
        exec(mm.pv.to_script(original, sources=replacements), {})
