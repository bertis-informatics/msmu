import warnings
from pathlib import Path
from types import NoneType
import json

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from . import label_info
from .normalise_sage_columns import normalise_sage_columns


class Reader:
    def __init__(self):
        md.set_options(pull_on_update=False)

    def split_desc_mtx(self, search_result: pd.DataFrame): ...

    #    def make_mudata(self, level, adata) -> md.MuData:
    #        mdata = md.MuData({level: adata})
    #


#        return mdata


class SageReader(Reader):
    def __init__(
        self,
        sage_output_dir: str | Path,
        sample_name: list[str],
        channel: list[str] | None,
        filename: list[str] | None,
    ) -> None:
        super().__init__()
        self._search_engine: str = "Sage"
        self._label: str | None = None
        self._sage_output_dir: Path = Path(sage_output_dir).absolute()
        self._sample_name: list[str] = sample_name

        self._channel: list[str] | None = channel
        self._filename: list[str] | None = filename

    def _read_sage_result(self) -> pd.DataFrame:
        sage_result_df = pd.read_csv(self._sage_result, sep="\t")
        sage_result_df = self._make_psm_index(data=sage_result_df)
        print(f"Sage psm result file: {sage_result_df.shape}")

        return sage_result_df

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = pd.read_csv(self._sage_quant, sep="\t")
        sage_quant_df = sage_quant_df.replace(0, np.nan)
        print(f"Sage quant result file: {sage_quant_df.shape}")

        return sage_quant_df

    def _read_sage_config(self) -> dict:
        sage_config = json.load(open(self._sage_json, "r"))

        return sage_config

    def _make_psm_index(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["filename_sub"] = df["filename"].apply(lambda x: ".".join(x.split(".")[:-1]))
        df["scan_num"] = df["scannr"].apply(lambda x: x.split("scan=")[1])

        df["psm_idx"] = df["filename_sub"] + "." + df["scan_num"]

        df = df.set_index("psm_idx", drop=True)
        df = df.rename_axis(index=None)

        df = df.drop(["filename_sub"], axis=1)

        return df

    def _rename_samples(self, sage_quant_df) -> pd.DataFrame:
        rename_dict: dict = self._make_raname_dict(sage_quant_df=sage_quant_df)
        print(rename_dict)
        renamed_sage_quant_df: pd.DataFrame = sage_quant_df.rename(columns=rename_dict)

        return renamed_sage_quant_df

    def _validate_sage_outputs(self) -> None:
        assert Path.exists(self._sage_result), f"{self._sage_result} does not exist!"
        assert Path.exists(self._sage_quant), f"{self._sage_quant} does not exist!"
        assert Path.exists(self._sage_json), f"{self._sage_json} does not exist!"

    def _get_sage_outputs(self) -> None:
        self._sage_result: Path = self._sage_output_dir / "results.sage.tsv"
        self._sage_quant: Path = self._sage_output_dir / f"{self._label}.tsv"
        self._sage_json: Path = self._sage_output_dir / "results.json"

    def _make_raname_dict(self, sage_quant_df) -> dict: ...

    def _import_sage(self):
        return pd.DataFrame(), pd.DataFrame()

    def _normalise_columns(self, sage_result_df):
        normalised_sage_result_df = normalise_sage_columns(sage_result_df=sage_result_df)

        return normalised_sage_result_df

    def _sage2mdata(self, sage_result_df, sage_quant_df, sage_config) -> md.MuData: ...

    def read(self) -> md.MuData:
        sage_result_df, sage_quant_df, sage_config = self._import_sage()
        mdata: md.MuData = self._sage2mdata(
            sage_result_df=sage_result_df, sage_quant_df=sage_quant_df, sage_config=sage_config
        )

        return mdata


class TmtSageReader(SageReader):
    def __init__(
        self,
        sage_output_dir,
        sample_name: list[str],
        channel: list[str] | None,
        filename: list[str] | None,
    ) -> None:
        super().__init__(
            sage_output_dir=sage_output_dir,
            sample_name=sample_name,
            channel=channel,
            filename=filename,
        )
        self._label: str | None = "tmt"

        self._get_sage_outputs()
        self._validate_sage_outputs()

    def _normalise_columns(self, sage_result_df) -> pd.DataFrame:
        return super()._normalise_columns(sage_result_df=sage_result_df)

    def _read_sage_result(self) -> pd.DataFrame:
        return super()._read_sage_result()

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = super()._read_sage_quant()
        sage_quant_df = self._make_psm_index(data=sage_quant_df)
        sage_quant_df = sage_quant_df.drop(["filename", "scannr", "ion_injection_time", "scan_num"], axis=1)

        return sage_quant_df

    def _rename_samples(self, sage_quant_df) -> pd.DataFrame:
        return super()._rename_samples(sage_quant_df)

    def _make_raname_dict(self, sage_quant_df) -> dict:
        plex: int = len(sage_quant_df.columns)
        tmt_labels: list = getattr(label_info, f"Tmt{plex}").label
        sage_labels: list = [f"tmt_{x}" for x in range(1, plex + 1)]

        channel_dict: dict = dict()
        for sage_col, tmt in zip(tmt_labels, sage_labels):
            channel_dict[sage_col] = tmt

        annotation_dict: dict = dict()
        if isinstance(self._channel, NoneType):
            print(
                '[WARNING] TMT Channels are not provied as argument "channel". Quantifiation columns will be renamed as an order of sample_name list.'
            )
            channel_list: list = tmt_labels
        else:
            channel_list: list = self._channel

        assert set(tmt_labels) == set(
            channel_list
        ), f"Provied Channel list is not matched to TMT{plex} channels. Diff_channel: {set(tmt_labels).difference(set(channel_list))}"

        for channel, sample_name in zip(channel_list, self._sample_name):
            annotation_dict[channel] = sample_name

        rename_dict = dict()
        for channel_key in channel_list:
            key_ = channel_dict[channel_key]
            rename_dict[key_] = annotation_dict[channel_key]

        return rename_dict

    def _import_sage(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        sage_result_df = self._read_sage_result()

        sage_quant_df = self._read_sage_quant()
        sage_quant_df = sage_quant_df.loc[sage_result_df.index,]
        sage_quant_df = self._rename_samples(sage_quant_df=sage_quant_df)

        sage_config = self._read_sage_config()

        return sage_result_df, sage_quant_df, sage_config

    def _sage2mdata(self, sage_result_df: pd.DataFrame, sage_quant_df: pd.DataFrame, sage_config: dict) -> md.MuData:
        adata = ad.AnnData(sage_quant_df.T)
        adata.var = self._normalise_columns(sage_result_df=sage_result_df)

        adata.varm["search_result"] = sage_result_df
        adata.uns["level"] = "psm"
        adata.uns["label"] = self._label
        adata.uns["sage_output_dir"] = str(self._sage_output_dir)
        adata.uns["sage_config"] = sage_config

        mdata = md.MuData({"psm": adata})

        return mdata

    def read(self) -> md.MuData:
        return super().read()


class LfqSageReader(SageReader):
    def __init__(self, sage_output_dir, sample_name, filename, channel) -> None:
        super().__init__(
            sage_output_dir=sage_output_dir,
            sample_name=sample_name,
            filename=filename,
            channel=channel,
        )
        self._label: str | None = "lfq"

        self._get_sage_outputs()
        self._validate_sage_outputs()

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = super()._read_sage_quant()
        sage_quant_df = sage_quant_df.set_index("peptide", drop=True)
        sage_quant_df = sage_quant_df.drop(["charge", "proteins", "q_value", "score", "spectral_angle"], axis=1)

        return sage_quant_df

    def _read_sage_result(self) -> pd.DataFrame:
        return super()._read_sage_result()

    def _import_sage(self):
        sage_result_df = self._read_sage_result()
        # column_normalised_sage_result_df = self._normalise_columns(sage_result_df)
        column_normalised_sage_result_df = sage_result_df.copy()

        sage_quant_df = self._read_sage_quant()
        sage_quant_df = self._rename_samples(sage_quant_df=sage_quant_df)

        sage_config = self._read_sage_config()

        return column_normalised_sage_result_df, sage_quant_df, sage_config

    def _rename_samples(self, sage_quant_df) -> pd.DataFrame:
        return super()._rename_samples(sage_quant_df)

    def _make_raname_dict(self, sage_quant_df) -> dict:
        filename_quant: list = sage_quant_df.columns.tolist()
        samples: list[str] = self._sample_name

        if isinstance(self._filename, NoneType):
            print(
                '[WARNING] filnames are not provied as argument "filename". Quantifiation columns will be renamed with in an order of sample_name list.'
            )
            filename: list[str] = filename_quant
        else:
            filename: list[str] = self._filename

        print(filename_quant)
        print(filename)

        # assert set(filename_quant) == set(
        #     filename
        # ), f"filenames in sage result and annotation are not matched. Diff filenames: {set(filename_quant).difference(set(filename))}"

        rename_dict: dict = dict()
        for f, sample in zip(filename, samples):
            rename_dict[f] = sample

        return rename_dict

    def _sage2mdata(self, sage_result_df, sage_quant_df, sage_config) -> md.MuData:
        empty_psm_mtx = pd.DataFrame(index=sage_result_df.index, columns=sage_quant_df.columns)
        adata_psm = ad.AnnData(empty_psm_mtx.T)
        adata_psm.var = self._normalise_columns(sage_result_df=sage_result_df)
        adata_psm.varm["search_result"] = sage_result_df
        adata_psm.uns["level"] = "psm"
        adata_psm.uns["label"] = self._label
        adata_psm.uns["search_output_dir"] = self._sage_output_dir
        adata_psm.uns["sage_config"] = sage_config

        adata_peptide = ad.AnnData(sage_quant_df.T)
        adata_peptide.uns["level"] = "peptide"

        mdata = md.MuData({"psm": adata_psm, "peptide": adata_peptide})

        return mdata

    def read(self) -> md.MuData:
        return super().read()
