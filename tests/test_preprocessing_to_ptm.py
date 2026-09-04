"""PTM site construction: what a site is named after, and what it does not depend on.

Sites are localised from the peptide's own accessions and the attached FASTA. That keeps the site id
a function of (peptide, FASTA) alone, so the same PTM data yields the same sites whether or not it
was processed alongside a global dataset -- and whichever global dataset that was.
"""

import inspect
import logging

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu as mm

SAMPLES = ["s1", "s2"]

# "SPGSPVLR" sits at offset 4 in P1 and offset 2 in P2, so the same peptide localises to a different
# residue number in each -- which is the whole reason a site id has to name its protein.
PROTEIN_SEQUENCES = {
    "P1": "MAAASPGSPVLRKKQ",
    "P2": "MMSPGSPVLRQQ",
    "P3": "MAAASPGSPVLRKKQ",
}
PHOSPHO = "[+79.97]"


def _make_peptide_mdata(peptide_rows: list[dict]) -> MuData:
    """Peptide modality with only what localisation needs -- no inferred protein_group."""
    var = pd.DataFrame(peptide_rows).set_index("peptide", drop=False)
    var.index.name = None
    values = np.arange(len(SAMPLES) * len(var), dtype=float).reshape(len(SAMPLES), len(var)) + 10.0
    adata = AnnData(X=values, obs=pd.DataFrame(index=SAMPLES), var=var)
    mdata = MuData({"peptide": adata})
    mdata.uns["protein_info"] = pd.DataFrame(
        {"Sequence": list(PROTEIN_SEQUENCES.values())}, index=list(PROTEIN_SEQUENCES)
    )
    return mdata


def _site_var(mdata: MuData) -> pd.DataFrame:
    return mdata["phospho_site"].var


def test_sites_are_localised_without_any_inferred_protein_group():
    """The headline: localisation needs an accession, a sequence and the FASTA -- nothing else.

    Protein grouping is a judgement made from one dataset's peptide evidence, so requiring it here
    would make site construction depend on a global dataset that may not exist yet.
    """
    mdata = _make_peptide_mdata(
        [{"peptide": f"SPGS{PHOSPHO}PVLR", "stripped_peptide": "SPGSPVLR", "proteins": "P1", "count_psm": 5}]
    )
    assert "protein_group" not in mdata["peptide"].var.columns

    result = mm.pp.to_ptm(mdata, modi_name="phospho", modification=PHOSPHO)

    assert list(result["phospho_site"].var_names) == ["P1|S8"]


def test_site_id_lists_accessions_flat_and_in_canonical_order():
    """Accessions are sorted, so the id does not depend on the order the engine listed them in.

    Two peptidoforms covering one site would otherwise disagree on its name and split into two
    features.
    """
    mdata = _make_peptide_mdata(
        [
            {"peptide": f"SPGS{PHOSPHO}PVLR", "stripped_peptide": "SPGSPVLR", "proteins": "P2;P1", "count_psm": 5},
        ]
    )

    result = mm.pp.to_ptm(mdata, modi_name="phospho", modification=PHOSPHO)

    # Flat ";" throughout -- no "," tier, because that tier only ever encoded a global grouping.
    assert list(result["phospho_site"].var_names) == ["P1|S8;P2|S6"]
    assert _site_var(result)["modified_protein"].tolist() == ["P1;P2"]


def test_accession_order_in_the_input_does_not_change_the_site_id():
    forward = mm.pp.to_ptm(
        _make_peptide_mdata(
            [{"peptide": f"SPGS{PHOSPHO}PVLR", "stripped_peptide": "SPGSPVLR", "proteins": "P1;P2", "count_psm": 5}]
        ),
        modi_name="phospho",
        modification=PHOSPHO,
    )
    reversed_order = mm.pp.to_ptm(
        _make_peptide_mdata(
            [{"peptide": f"SPGS{PHOSPHO}PVLR", "stripped_peptide": "SPGSPVLR", "proteins": "P2;P1", "count_psm": 5}]
        ),
        modi_name="phospho",
        modification=PHOSPHO,
    )

    assert list(forward["phospho_site"].var_names) == list(reversed_order["phospho_site"].var_names)


def test_psm_count_is_not_multiplied_by_the_number_of_accessions():
    """Exploding over accessions duplicates the peptidoform's row; summing there would inflate it."""
    mdata = _make_peptide_mdata(
        [{"peptide": f"SPGS{PHOSPHO}PVLR", "stripped_peptide": "SPGSPVLR", "proteins": "P1;P2;P3", "count_psm": 5}]
    )

    result = mm.pp.to_ptm(mdata, modi_name="phospho", modification=PHOSPHO)

    assert _site_var(result)["count_psm"].tolist() == [5]


def test_a_doubly_modified_peptidoform_reports_both_of_its_sites():
    """A species phosphorylated at two residues is part of the abundance of each of them."""
    mdata = _make_peptide_mdata(
        [
            {
                "peptide": f"S{PHOSPHO}PGS{PHOSPHO}PVLR",
                "stripped_peptide": "SPGSPVLR",
                "proteins": "P1",
                "count_psm": 5,
            }
        ]
    )

    result = mm.pp.to_ptm(mdata, modi_name="phospho", modification=PHOSPHO)

    assert sorted(result["phospho_site"].var_names) == ["P1|S5", "P1|S8"]


def test_median_polish_is_the_default_rollup():
    """It models a per-peptidoform effect, so a site's value does not move when the set of
    peptidoforms supporting it changes between samples. For a single-peptidoform site it is
    identical to median, which is the majority case -- so the default costs nothing there."""
    assert inspect.signature(mm.pp.to_ptm).parameters["agg_method"].default == "median_polish"


def _captured_msmu_warnings() -> tuple[logging.Handler, list[logging.LogRecord]]:
    """msmu's package logger sets propagate=False, so pytest's caplog (a root handler) sees nothing.

    Attach directly to the logger the module actually writes to instead.
    """
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    return _Collector(level=logging.WARNING), records


def test_linear_intensities_are_flagged_before_an_additive_rollup():
    mdata = _make_peptide_mdata(
        [{"peptide": f"SPGS{PHOSPHO}PVLR", "stripped_peptide": "SPGSPVLR", "proteins": "P1", "count_psm": 5}]
    )
    mdata["peptide"].X = mdata["peptide"].X * 1e6

    handler, records = _captured_msmu_warnings()
    summarisation_logger = logging.getLogger("msmu._preprocessing._summarisation")
    summarisation_logger.addHandler(handler)
    try:
        mm.pp.to_ptm(mdata, modi_name="phospho", modification=PHOSPHO, agg_method="median_polish")
    finally:
        summarisation_logger.removeHandler(handler)

    assert any("log2" in record.getMessage() for record in records)


def test_missing_accession_column_names_what_is_required():
    mdata = _make_peptide_mdata(
        [{"peptide": f"SPGS{PHOSPHO}PVLR", "stripped_peptide": "SPGSPVLR", "proteins": "P1", "count_psm": 5}]
    )
    del mdata["peptide"].var["proteins"]

    with pytest.raises(ValueError, match="proteins"):
        mm.pp.to_ptm(mdata, modi_name="phospho", modification=PHOSPHO)
