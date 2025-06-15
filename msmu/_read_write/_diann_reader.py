import re
from pathlib import Path
from types import NoneType

import anndata as ad
import mudata as md
import pandas as pd

from ._sage_reader import Reader


class DiannReader(Reader):
    def __init__(
        self,
        diann_output_dir: str | Path,
        sample_name: list[str],
        filename: list[str] | None = None,
        fasta: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._search_engine = "diann"
        self._label = "lfq"
        self._diann_output_dir = Path(diann_output_dir).absolute()
        self._sample_name = sample_name
        self._filename = filename
        self._fasta = fasta

        self._get_diann_outputs()
        self._validate_diann_outputs()

    def _get_diann_outputs(self) -> None:
        self._diann_result = self._diann_output_dir / "report.tsv"

    def _validate_diann_outputs(self) -> None:
        if not self._diann_result.exists():
            raise FileNotFoundError(f"{self._diann_result} does not exist!")

    def _read_file(self, file_path: Path, sep: str = "\t") -> pd.DataFrame:
        return pd.read_csv(file_path, sep=sep)

    def _make_precursor_index(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["precursor_idx"] = df["Run"] + "." + df["MS2.Scan"].astype(str) + "." + df["Precursor.Id"]

        return df

    def _make_diann_quant(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = df[["precursor_idx", "Run", "Precursor.Quantity"]]
        df = df.pivot(index="precursor_idx", columns="Run", values="Precursor.Quantity")
        df = df.rename_axis(index=None, columns=None)

        return df

    def _get_mbr(self, data: pd.DataFrame) -> None:
        df = data.copy()
        mbr_cols: list = [x for x in df.columns if x.startswith("Lib")]
        if len(mbr_cols) > 0:
            self._mbr: bool = True
        else:
            self._mbr: bool = False

    def _import_diann(self):
        diann_result_df = self._read_file(self._diann_result)
        # TODO: read Diann Config from report.log.txt (or from cfg)
        #        with open(self._diann_output_dir / "report.log.txt", "r") as f:
        #            header = f.readline().strip().split("\t")

        return diann_result_df

    def _make_rename_dict(self, diann_result_df: pd.DataFrame) -> dict:
        if isinstance(self._filename, NoneType):
            print("[WARNING] No filename provided. Using sample name as filename.")

        filenames = self._filename or diann_result_df["Run"].unique().tolist()
        sample_names = self._sample_name or diann_result_df["Run"].unique().tolist()
        return {filename: sample for filename, sample in zip(filenames, sample_names)}

    def _normalise_columns(self, diann_result_df: pd.DataFrame) -> pd.DataFrame:
        return normalise_diann_columns(diann_result_df)

    def _diann2mdata(self, diann_result_df: pd.DataFrame, diann_quant_df: pd.DataFrame) -> md.MuData:
        rename_dict = self._make_rename_dict(diann_result_df)
        diann_quant_df = diann_quant_df.rename(columns=rename_dict)

        adata_precursor = ad.AnnData(diann_quant_df.T.astype("float"))
        normalised_diann_result_df = self._normalise_columns(diann_result_df)
        normalised_diann_result_df = normalised_diann_result_df.loc[diann_quant_df.index, :]
        adata_precursor.var = normalised_diann_result_df
        varm_df = diann_result_df.set_index("precursor_idx")
        varm_df = varm_df.rename_axis(index=None)
        varm_df = varm_df.loc[diann_quant_df.index]
        adata_precursor.varm["search_result"] = varm_df
        adata_precursor.uns.update(
            {
                "level": "precursor",
                "search_engine": self._search_engine,
                "label": self._label,
                "search_output_dir": str(self._diann_output_dir),
                "search_config": "should be a config file",
            }
        )
        mdata: md.MuData = md.MuData({"feature": adata_precursor})
        # mdata: md.MuData = self._add_obs_tag(mdata, rename_dict)
        mdata.update_obs()

        return mdata

    def read(self):
        diann_result_df = self._import_diann()
        diann_result_df = self._make_precursor_index(diann_result_df)
        diann_quant_df = self._make_diann_quant(diann_result_df)
        # self._get_mbr(diann_result_df)

        mdata: md.MuData = self._diann2mdata(diann_result_df, diann_quant_df)
        mdata = self._assign_protein_id_info(mdata=mdata, fasta=self._fasta)

        return mdata


#
def rename_diann_columns(diann_result_df: pd.DataFrame, mbr: bool = True) -> pd.DataFrame:
    """
    Renames columns in the DIANN result DataFrame to a more user-friendly format.

    Args:
        diann_result_df (pd.DataFrame): Input DataFrame with DIANN results.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    if mbr:
        q_value_prefix = "Lib"
    else:
        q_value_prefix = "Global"

    rename_dict = {
        "Protein.Ids": "proteins",
        "Modified.Sequence": "peptide",
        "Stripped.Sequence": "stripped_peptide",
        "Run": "filename",
        "MS2.Scan": "scan_num",
        "Precursor.Charge": "charge",
        f"{q_value_prefix}.Q.Value": "spectrum_q",
        f"{q_value_prefix}.Peptidoform.Q.Value": "peptide_q",
        f"{q_value_prefix}.PG.Q.Value": "protein_q",
    }

    return diann_result_df.rename(columns=rename_dict)


def normalise_diann_columns(diann_result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the DIANN result DataFrame by selecting relevant columns and adding derived columns.

    Args:
        diann_result_df (pd.DataFrame): Input DataFrame with DIANN results.

    Returns:
        pd.DataFrame: Normalized DataFrame with additional columns.
    """
    used_cols = [
        "precursor_idx",
        "proteins",
        "peptide",
        "stripped_peptide",
        "filename",
        "scan_num",
        "charge",
        "spectrum_q",
        "peptide_q",
        "protein_q",
    ]
    normalised_diann_result_df: pd.DataFrame = rename_diann_columns(diann_result_df)
    used_cols = [x for x in used_cols if x in normalised_diann_result_df.columns]
    normalised_diann_result_df = normalised_diann_result_df[used_cols].copy()
    normalised_diann_result_df = normalised_diann_result_df.set_index("precursor_idx")
    normalised_diann_result_df = normalised_diann_result_df.rename_axis(index=None)

    # calculate missed cleavages
    normalised_diann_result_df["missed_cleavages"] = normalised_diann_result_df.apply(
        lambda x: count_missed_cleavages(x["peptide"], "trypsin"), axis=1
    )

    return normalised_diann_result_df


def count_missed_cleavages(peptide: str, enzyme: str) -> int:
    """
    Count missed cleavages for trypsin in a given peptide sequence.
    Trypsin cleaves after K or R, except when followed by P.
    """
    if enzyme != "trypsin":
        raise ValueError("This function only supports trypsin enzyme.")
    # Find all cleavage sites in the full (ideal) digest
    cleavage_sites = [m.start() + 1 for m in re.finditer(r"(?<=[KR])(?!P)", peptide)]

    # Each cleavage site that is still present in the peptide indicates a missed cleavage
    return len(cleavage_sites)
