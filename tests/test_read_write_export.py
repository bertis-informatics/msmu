from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from msmu._read_write._export import to_readable, write_flashlfq_input


def _make_psm_mdata() -> MuData:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    obs = pd.DataFrame(index=["s1", "s2"])
    var = pd.DataFrame(
        {
            "filename": ["f1.raw", "f2.raw"],
            "rt": [10.0, 20.0],
            "charge": [2, 3],
            "stripped_peptide": ["AA", "BB"],
            "peptide": ["AA", "BB"],
            "calcmass": [100.0, 200.0],
            "proteins": ["P1", "P2"],
            "extra": [1, 2],
        },
        index=["f1", "f2"],
    )
    adata = AnnData(x, obs=obs, var=var)
    return MuData({"psm": adata})


def test_to_readable_include_exclude_and_quant():
    mdata = _make_psm_mdata()
    df = to_readable(mdata, modality="psm", include=["filename", "rt"], quantification=False)
    assert df.columns.tolist() == ["filename", "rt"]

    df2 = to_readable(mdata, modality="psm", exclude="extra", quantification=True)
    assert "extra" not in df2.columns
    assert "s1" in df2.columns


def test_write_flashlfq_input(tmp_path):
    mdata = _make_psm_mdata()
    output = Path(tmp_path) / "flashlfq.tsv"
    write_flashlfq_input(mdata, output)
    content = output.read_text().splitlines()
    assert "File Name" in content[0]
    assert "Protein Accession" in content[0]
