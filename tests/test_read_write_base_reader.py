from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msmu._read_write._base_reader import SearchResultReader, SearchResultSettings


class DummyReader(SearchResultReader):
    def __init__(
        self,
        identification_df: pd.DataFrame,
        quant_df: pd.DataFrame | None = None,
        *,
        ident_quant_merged: bool = False,
        quant_level: str | None = None,
        has_decoy: bool = True,
    ):
        super().__init__()
        self._identification_df = identification_df
        self._quant_df = quant_df
        self._feature_rename_dict = {
            "Protein": "proteins",
            "Peptide": "peptide",
            "Stripped": "stripped_peptide",
            "Filename": "filename",
            "ScanNum": "scan_num",
            "Charge": "charge",
            "PeptideLength": "peptide_length",
            "Decoy": "decoy",
        }
        self._cols_to_stringify = ["ScanNum"]
        self.used_feature_cols = self.used_feature_cols + ["decoy"]
        self.search_settings = SearchResultSettings(
            search_engine="dummy",
            quantification=None if quant_level is None else "dummy",
            label=None,
            acquisition=None,
            identification_file=Path("dummy.csv"),
            identification_level="psm",
            quantification_file=Path("dummy_quant.csv") if quant_level is not None else None,
            quantification_level=quant_level,
            ident_quant_merged=ident_quant_merged,
            has_decoy=has_decoy,
        )

    def _import_search_results(self) -> dict:
        return {
            "identification": self._identification_df,
            "quantification": self._quant_df,
        }

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        return identification_df

    def _split_merged_identification_quantification(
        self, identification_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._quant_df is None:
            raise ValueError("quant_df is required when ident/quant are merged")
        return identification_df, self._quant_df


@pytest.fixture
def identification_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Protein": ["P1", "P2"],
            "Peptide": ["AA", "BB"],
            "Stripped": ["AA", "BB"],
            "Filename": ["file1.raw", "file2.raw"],
            "ScanNum": [1, 2],
            "Charge": [2, 3],
            "PeptideLength": [2, 2],
            "Decoy": [0, 1],
        }
    )


def test_stringify_cols_casts_to_str(identification_df: pd.DataFrame):
    reader = DummyReader(identification_df)
    out = reader._stringify_cols(identification_df)
    assert out["ScanNum"].dtype == object


def test_make_mudata_input_feature_only(identification_df: pd.DataFrame):
    reader = DummyReader(identification_df, quant_df=None, quant_level=None, has_decoy=True)
    mudata_input = reader._make_mudata_input()
    assert mudata_input.norm_identification_df.index.tolist() == ["file1.raw.1"]
    assert mudata_input.decoy_df is not None
    assert mudata_input.norm_quant_df is None


def test_build_mudata_feature_only(identification_df: pd.DataFrame):
    reader = DummyReader(identification_df, quant_df=None, quant_level=None, has_decoy=True)
    mudata_input = reader._make_mudata_input()
    mdata = reader._build_mudata(mudata_input)
    assert "psm" in mdata.mod_names
    assert mdata["psm"].var.index.tolist() == ["file1.raw.1"]
    assert "decoy" in mdata["psm"].uns_keys()


def test_build_mudata_with_quantification(identification_df: pd.DataFrame):
    quant_df = pd.DataFrame(
        {
            "s1": [1.0, 2.0],
            "s2": [0.0, 3.0],
        },
        index=["file1.raw.1", "file2.raw.2"],
    )
    reader = DummyReader(
        identification_df,
        quant_df=quant_df,
        quant_level="psm",
        has_decoy=True,
    )
    mudata_input = reader._make_mudata_input()
    mdata = reader._build_mudata(mudata_input)
    assert mdata["psm"].X.shape == (2, 1)
    assert np.isnan(mdata["psm"].X[1, 0])
