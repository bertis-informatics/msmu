import json
import warnings
from pathlib import Path
from types import NoneType

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
        channel: list[str] | None = None,
        filename: list[str] | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__()
        self._search_engine = "Sage"
        self._label = label
        self._sage_output_dir = Path(sage_output_dir).absolute()
        self._sample_name = sample_name
        self._channel = channel
        self._filename = filename

        self._get_sage_outputs()
        self._validate_sage_outputs()

    def _get_sage_outputs(self) -> None:
        self._sage_result = self._sage_output_dir / "results.sage.tsv"
        self._sage_quant = self._sage_output_dir / f"{self._label}.tsv"
        self._sage_json = self._sage_output_dir / "results.json"

    def _validate_sage_outputs(self) -> None:
        for file_path in [self._sage_result, self._sage_quant, self._sage_json]:
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path} does not exist!")

    def _read_file(self, file_path: Path, sep: str = "\t") -> pd.DataFrame:
        return pd.read_csv(file_path, sep=sep)

    def _read_sage_result(self) -> pd.DataFrame:
        sage_result_df = pd.read_csv(self._sage_result, sep="\t")
        sage_result_df = self._make_psm_index(data=sage_result_df)
        print(f"Sage result file loaded: {sage_result_df.shape}")

        return sage_result_df

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = pd.read_csv(self._sage_quant, sep="\t").replace(0, np.nan)
        print(f"Sage quant file loaded: {sage_quant_df.shape}")
        return sage_quant_df

    def _read_sage_config(self) -> dict:
        with open(self._sage_json, "r") as f:
            return json.load(f)

    def _make_psm_index(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["filename_sub"] = df["filename"].str.rsplit(".", n=1).str[0]
        df["scan_num"] = df["scannr"].str.split("scan=").str[1]
        df["psm_idx"] = df["filename_sub"] + "." + df["scan_num"]
        df = df.set_index("psm_idx", drop=True).rename_axis(index=None)
        return df.drop(["filename_sub"], axis=1)

    def _rename_samples(self, sage_quant_df: pd.DataFrame) -> pd.DataFrame:
        rename_dict = self._make_rename_dict(sage_quant_df)
        return sage_quant_df.rename(columns=rename_dict)

    def _make_rename_dict(self, sage_quant_df: pd.DataFrame) -> dict:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _import_sage(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _normalise_columns(
        self, sage_result_df: pd.DataFrame, precursor_charge: bool = False
    ) -> pd.DataFrame:
        return normalise_sage_columns(
            sage_result_df=sage_result_df, precursor_charge=precursor_charge
        )

    def _add_obs_tag(self, mdata: md.MuData, rename_dict: dict) -> md.MuData:
        mdata.obs["tag"] = mdata.obs.index.map(
            {sample: tag for tag, sample in rename_dict.items()}
        )
        return mdata

    def _sage2mdata(
        self,
        sage_result_df: pd.DataFrame,
        sage_quant_df: pd.DataFrame,
        sage_config: dict,
    ) -> md.MuData:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def read(self) -> md.MuData:
        sage_result_df, sage_quant_df, sage_config = self._import_sage()
        return self._sage2mdata(sage_result_df, sage_quant_df, sage_config)


class TmtSageReader(SageReader):
    def __init__(
        self,
        sage_output_dir: str | Path,
        sample_name: list[str],
        channel: list[str] | None = None,
        filename: list[str] | None = None,
    ) -> None:
        super().__init__(sage_output_dir, sample_name, channel, filename, label="tmt")

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = super()._read_sage_quant()
        sage_quant_df = self._make_psm_index(data=sage_quant_df)
        sage_quant_df = sage_quant_df.drop(
            ["filename", "scannr", "ion_injection_time", "scan_num"], axis=1
        )

        return sage_quant_df

    def _make_rename_dict(self, sage_quant_df: pd.DataFrame) -> dict:
        if isinstance(self._channel, NoneType):
            print(
                '[WARNING] TMT Channels are not provied as argument "channel". Quantifiation columns will be renamed as an order of sample_name list.'
            )

        plex = len(sage_quant_df.columns)
        tmt_labels = getattr(label_info, f"Tmt{plex}").label
        sage_labels = [f"tmt_{x}" for x in range(1, plex + 1)]
        channel_list = self._channel or sage_labels

        if set(tmt_labels) != set(channel_list):
            raise ValueError(
                f"Provied Channel list is not matched to TMT{plex} channels. Diff_channel: {set(tmt_labels).difference(set(channel_list))}"
            )

        channel_dict = {sage_col: tmt for sage_col, tmt in zip(tmt_labels, sage_labels)}
        annotation_dict = {
            channel: sample for channel, sample in zip(channel_list, self._sample_name)
        }

        return {channel_dict[key]: annotation_dict[key] for key in channel_list}

    def _import_sage(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        sage_result_df = self._read_sage_result()

        sage_quant_df = self._read_sage_quant()
        sage_quant_df = sage_quant_df.loc[sage_result_df.index,]

        sage_config = self._read_sage_config()

        return sage_result_df, sage_quant_df, sage_config

    def _sage2mdata(
        self,
        sage_result_df: pd.DataFrame,
        sage_quant_df: pd.DataFrame,
        sage_config: dict,
    ) -> md.MuData:
        rename_dict = self._make_rename_dict(sage_quant_df)
        adata = ad.AnnData(sage_quant_df.rename(columns=rename_dict).T)
        adata.var = self._normalise_columns(sage_result_df)
        adata.varm["search_result"] = sage_result_df
        adata.uns.update(
            {
                "level": "psm",
                "label": self._label,
                "sage_output_dir": str(self._sage_output_dir),
                "sage_config": sage_config,
            }
        )
        mdata: md.MuData = md.MuData({"psm": adata})
        mdata: md.MuData = self._add_obs_tag(mdata, rename_dict)
        mdata.update_obs()

        return mdata


class LfqSageReader(SageReader):
    def __init__(
        self,
        sage_output_dir: str | Path,
        sample_name: list[str],
        channel: list[str] | None = None,
        filename: list[str] | None = None,
    ) -> None:
        super().__init__(sage_output_dir, sample_name, channel, filename, label="lfq")

    def _read_sage_quant(self) -> pd.DataFrame:
        sage_quant_df = super()._read_sage_quant()

        # make precursor ID
        sage_quant_df.loc[:, "peptide"] = sage_quant_df.apply(
            lambda x: x["peptide"] + "." + str(x["charge"]), axis=1
        )

        sage_quant_df = sage_quant_df.set_index("peptide", drop=True)
        self._precursor_charge = (
            True if sage_quant_df["charge"].unique()[0] != -1 else False
        )
        sage_quant_df = sage_quant_df.drop(
            ["charge", "proteins", "q_value", "score", "spectral_angle"], axis=1
        )

        return sage_quant_df

    def _import_sage(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        sage_result_df = self._read_sage_result()

        sage_quant_df = self._read_sage_quant()

        sage_config = self._read_sage_config()

        return sage_result_df, sage_quant_df, sage_config

    def _make_rename_dict(self, sage_quant_df: pd.DataFrame) -> dict:
        if isinstance(self._filename, NoneType):
            print(
                '[WARNING] filnames are not provied as argument "filename". Quantifiation columns will be renamed with in an order of sample_name list.'
            )

        filenames = self._filename or sage_quant_df.columns.tolist()
        return {
            filename: sample for filename, sample in zip(filenames, self._sample_name)
        }

    def _sage2mdata(
        self,
        sage_result_df: pd.DataFrame,
        sage_quant_df: pd.DataFrame,
        sage_config: dict,
    ) -> md.MuData:
        rename_dict = self._make_rename_dict(sage_quant_df)
        sage_quant_df = sage_quant_df.rename(columns=rename_dict)

        adata_psm = ad.AnnData(
            pd.DataFrame(index=sage_result_df.index, columns=sage_quant_df.columns).T
        )
        adata_psm.var = self._normalise_columns(sage_result_df, self._precursor_charge)
        adata_psm.varm["search_result"] = sage_result_df
        adata_psm.uns.update(
            {
                "level": "psm",
                "label": self._label,
                "sage_output_dir": str(self._sage_output_dir),
                "sage_config": sage_config,
            }
        )

        adata_peptide = ad.AnnData(sage_quant_df.T)
        adata_peptide.uns["level"] = "peptide"

        mdata = md.MuData({"psm": adata_psm, "peptide": adata_peptide})
        mdata = self._add_obs_tag(mdata=mdata, rename_dict=rename_dict)
        mdata.update_obs()

        return mdata
