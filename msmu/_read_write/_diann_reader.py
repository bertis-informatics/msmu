from pathlib import Path

import pandas as pd

from ._sage_reader import Reader


class DiannReader(Reader):
    def __init__(
        self,
        diann_output_dir: str | Path,
        sample_name: list[str],
        filename: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._search_engine = "diann"
        self._label = "lfq"
        self._diann_output_dir = Path(diann_output_dir).absolute()
        self._sample_name = sample_name
        self._filename = filename

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
        df["precursor_idx"] = (
            df["Run"] + "." + df["MS2.Scan"] + "." + df["Precursor.Id"]
        )
        df = df.set_index(df, "precursor_idx")

        return df

    def _make_diann_quant(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = df[["precursor_idx", "Run", "Precursor.Quantity"]]
        df = df.pivot(index="Precursor.Id", columns="Run", values="Precursor.Quantity")
        df = df.rename_axis(index=None, columns=None)
        df = df.set_index(df, "precursor.Id")

        return df

    def _get_mbr(self, data: pd.DataFrame) -> None:
        df = data.copy()
        mbr_cols: list = [x for x in df.columns if "Global" in x]
        if len(mbr_cols) > 0:
            self._mbr: bool = True
        else:
            self._mbr: bool = False

    def read(self): ...

    def _import_diann(self):
        diann_result_df = self._read_file(self._diann_result)
        return diann_result_df

    def _diann2mdata(
        self, diann_result_df: pd.DataFrame, diann_quant_df: pd.DataFrame
    ): ...
