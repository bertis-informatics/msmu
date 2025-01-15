import mudata as md
import anndata as ad
from pathlib import Path
import pandas as pd
import numpy as np

from . import label_info


class Reader:
    def split_desc_mtx(self, search_result: pd.DataFrame): ...

    #    def make_mudata(self, level, adata) -> md.MuData:
    #        mdata = md.MuData({level: adata})
    #


#        return mdata


class SageReader(Reader):
    def __init__(self, sage_output_dir: str | Path, annotation) -> None:
        super().__init__()
        self._search_engine: str = "Sage"
        self._label: str | None = None
        self._sage_output_dir: Path = Path(sage_output_dir).absolute()
        self._annotation = annotation

        self.desc_cols: list = list()

    def _read_sage_result(self) -> pd.DataFrame:
        sage_result_df = pd.read_csv(self._sage_result, sep="\t")

        sage_result_df = self._make_psm_index(data=sage_result_df)

        return sage_result_df

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = pd.read_csv(self._sage_quant, sep="\t")
        sage_quant_df = sage_quant_df.replace(0, np.nan)

        return sage_quant_df

    def _make_psm_index(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["filename_sub"] = df["filename"].apply(lambda x: ".".join(x.split(".")[:-1]))
        df["scan_num"] = df["scannr"].apply(lambda x: x.split("scan=")[1])

        df["psm_idx"] = df["filename_sub"] + "." + df["scan_num"]

        df = df.set_index("psm_idx", drop=True)
        df = df.rename_axis(index=None)

        df = df.drop(["filename_sub", "scan_num"], axis=1)

        return df

    def _rename_samples(self, sage_quant_df) -> pd.DataFrame:
        rename_dict: dict = self._make_raname_dict(
            sage_quant_df=sage_quant_df, annotation=self._annotation
        )
        renamed_sage_quant_df: pd.DataFrame = sage_quant_df.rename(columns=rename_dict)

        return renamed_sage_quant_df

    def _validate_sage_outputs(self) -> None:
        assert Path.exists(self._sage_result), f"{self._sage_result} is not exists!"
        assert Path.exists(self._sage_quant), f"{self._sage_quant} is not exists!"

    def _get_sage_outputs(self) -> None:
        self._sage_result: Path = self._sage_output_dir / "results.sage.tsv"
        self._sage_quant: Path = self._sage_output_dir / f"{self._label}.tsv"

    def _make_raname_dict(self, sage_quant_df, annotation) -> dict: ...

    def _import_sage(self):
        return pd.DataFrame(), pd.DataFrame()

    def _sage2mdata(self, sage_result_df, sage_quant_df) -> md.MuData: ...

    def read(self) -> md.MuData:
        sage_result_df, sage_quant_df = self._import_sage()
        mdata: md.MuData = self._sage2mdata(
            sage_result_df=sage_result_df, sage_quant_df=sage_quant_df
        )

        return mdata


class TmtSageReader(SageReader):
    def __init__(self, sage_output_dir, annotation) -> None:
        super().__init__(sage_output_dir=sage_output_dir, annotation=annotation)
        self._label: str | None = "Tmt"

        self._get_sage_outputs()
        self._validate_sage_outputs()

    def _sage2adata(self) -> ad.AnnData: ...

    def _normalise_columns(self) -> pd.DataFrame: ...

    def _read_sage_result(self) -> pd.DataFrame:
        return super()._read_sage_result()

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = super()._read_sage_quant()
        sage_quant_df = self._make_psm_index(data=sage_quant_df)
        sage_quant_df = sage_quant_df.drop(
            ["filename", "scannr", "ion_injection_time"], axis=1
        )

        return sage_quant_df

    def _rename_samples(self, sage_quant_df) -> pd.DataFrame:
        return super()._rename_samples(sage_quant_df)

    def _make_raname_dict(self, sage_quant_df, annotation) -> dict:
        plex: int = len(sage_quant_df.columns)
        tmt_labels: list = getattr(label_info, f"Tmt{plex}").label
        sage_labels: list = [f"tmt_{x}" for x in range(1, plex + 1)]

        annotation_tmt = list(annotation["tmt"])
        annotation_sample = list(annotation["sample"])

        assert tmt_labels == annotation_tmt, "TMT label is not matched."
        rename_dict: dict[str, str] = dict()
        for tmt, sample in zip(sage_labels, annotation_sample):
            rename_dict[tmt] = sample

        return rename_dict

    def _import_sage(self):
        sage_result_df = self._read_sage_result()
        # column_normalised_sage_result_df = self._normalise_columns(sage_result_df)
        column_normalised_sage_result_df = sage_result_df.copy()

        sage_quant_df = self._read_sage_quant()
        sage_quant_df = sage_quant_df.loc[column_normalised_sage_result_df.index,]
        sage_quant_df = self._rename_samples(sage_quant_df=sage_quant_df)

        return column_normalised_sage_result_df, sage_quant_df

    def _sage2mdata(self, sage_result_df, sage_quant_df) -> md.MuData:
        adata = ad.AnnData(sage_quant_df.T)
        adata.var = sage_result_df[
            ["proteins", "peptide", "spectrum_q", "peptide_q", "protein_q"]
        ]
        adata.varm["search_result"] = sage_result_df
        adata.uns["level"] = "psm"
        adata.uns["label"] = self._label

        mdata = md.MuData({"psm": adata})

        return mdata

    def read(self) -> md.MuData:
        return super().read()


class LfqSageReader(SageReader):
    def __init__(self, sage_output_dir, annotation) -> None:
        super().__init__(sage_output_dir=sage_output_dir, annotation=annotation)
        self._label: str | None = "lfq"

        self._get_sage_outputs()
        self._validate_sage_outputs()

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = super()._read_sage_quant()
        sage_quant_df = sage_quant_df.set_index("peptide", drop=True)
        sage_quant_df = sage_quant_df.drop(
            ["charge", "proteins", "q_value", "score", "spectral_angle"], axis=1
        )

        return sage_quant_df

    def _read_sage_result(self) -> pd.DataFrame:
        return super()._read_sage_result()

    def _import_sage(self):
        sage_result_df = self._read_sage_result()
        # column_normalised_sage_result_df = self._normalise_columns(sage_result_df)
        column_normalised_sage_result_df = sage_result_df.copy()

        sage_quant_df = self._read_sage_quant()
        sage_quant_df = self._rename_samples(sage_quant_df=sage_quant_df)

        return column_normalised_sage_result_df, sage_quant_df

    def _rename_samples(self, sage_quant_df) -> pd.DataFrame:
        return super()._rename_samples(sage_quant_df)

    def _make_raname_dict(self, sage_quant_df, annotation) -> dict:
        filename_quant: list = sage_quant_df.columns.tolist()
        filename_annotation: list = annotation["filename"]
        samples: list = annotation["sample"]

        assert (
            filename_quant == filename_annotation
        ), "filenames in sage result and annotation are not matched"
        rename_dict: dict = dict()
        for filename, sample in zip(filename_annotation, samples):
            rename_dict[filename] = sample

        return rename_dict

    def _sage2mdata(self, sage_result_df, sage_quant_df) -> md.MuData:
        empty_psm_mtx = pd.DataFrame(
            index=sage_result_df.index, columns=sage_quant_df.columns
        )
        adata_psm = ad.AnnData(empty_psm_mtx.T)
        adata_psm.var = sage_result_df[
            ["proteins", "peptide", "spectrum_q", "peptide_q", "protein_q"]
        ]
        adata_psm.varm["search_result"] = sage_result_df
        adata_psm.uns["level"] = "psm"
        adata_psm.uns["label"] = self._label

        adata_peptide = ad.AnnData(sage_quant_df.T)
        adata_peptide.uns["level"] = "peptide"

        mdata = md.MuData({"psm": adata_psm, "peptide": adata_peptide})

        return mdata

    def read(self) -> md.MuData:
        return super().read()


#
#
class DiannReader(Reader):
    def __init__(self) -> None:
        super().__init__()
        self._search_engine = "Diann"


#
def read_sage(
    sage_output_dir: str | Path,
    annotation: dict | pd.DataFrame | str | Path,
    label: str,
) -> md.MuData:
    if label == "Tmt":
        reader_cls = TmtSageReader
    elif label == "lfq":
        reader_cls = LfqSageReader
    else:
        raise ValueError("Argument label should be one of 'Tmt', 'lfq'.")

    reader = reader_cls(sage_output_dir=sage_output_dir, annotation=annotation)
    mdata = reader.read()

    return mdata


# TODO: NEEDED MAIN FUNCTIONS
def merge_mudata(): ...


def make_sample_annotation(): ...


# TODO: MARK IS_BLANK and IS_IRS function for TMT
