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

    # def split_desc_mtx(self, search_result: pd.DataFrame): ...

    #    def make_mudata(self, level, adata) -> md.MuData:
    #        mdata = md.MuData({level: adata})
    #


#        return mdata

class ProteinIdParser:
    def _parse_uniprot_accession(self, proteins:pd.Series) -> pd.DataFrame:
        protein_df:pd.DataFrame = pd.DataFrame(proteins)
        protein_df['index'] = range(len(protein_df))
        protein_df['protein'] = protein_df['proteins'].apply(lambda x: x.split(';'))
        protein_df = protein_df.explode('protein')

        uniprot_id_category:list = ['source', 'accession', 'protein_name']
        for idx, cat_ in enumerate(uniprot_id_category):
            protein_df[cat_] = protein_df['protein'].apply(lambda x: x.split("|")[idx])
        
        protein_df['accession'] = protein_df.apply(lambda x: f"rev_{x['accession']}" if x['protein'].startswith('rev_') else x['accession'], axis=1)
        protein_df['accession'] = protein_df.apply(lambda x: f"contam_{x['accession']}" if x['protein'].startswith('contam_') else x['accession'], axis=1)
        
        return protein_df

    def _make_protein_info(self, protein_df:pd.DataFrame) -> pd.DataFrame:
        protein_info:pd.DataFrame = protein_df.copy()

        protein_info = protein_info.drop_duplicates('accession')
        protein_info = protein_info.drop(columns=["index", "proteins"])

        protein_info = protein_info.loc[protein_info['source'].str.startswith('rev_') == False, ]
        protein_info = protein_info.loc[protein_info['source'].str.startswith('contam_') == False, ]

        protein_info = protein_info.sort_values("accession")
        protein_info = protein_info.reset_index(drop=True)
        
        return protein_info

    def parse(self, proteins:pd.Series, source:str="uniprot"):
        if source == 'uniprot':
            protein_df:pd.DataFrame = self._parse_uniprot_accession(proteins=proteins)
            protein_df_grouped = protein_df.groupby(['index', 'proteins'], as_index = False).agg(';'.join)
            protein_df_grouped = protein_df_grouped.sort_values('index')

            self.accession:list[str] = protein_df_grouped['accession'].tolist()
        else:
            raise NotImplementedError('For now, protein parse only can be applied to uniprot fasta')

        self.protein_info:pd.DataFrame = self._make_protein_info(protein_df=protein_df)


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
    
    def _assign_protein_id_info(self, mdata:md.MuData) -> md.MuData:
        protein_id_info:ProteinIdParser = ProteinIdParser()
        protein_id_info.parse(proteins=mdata['psm'].var['proteins'])
        
        mdata.mod['psm'].var['proteins'] = protein_id_info.accession
        mdata.uns['protein_info'] = protein_id_info.protein_info

        return mdata

    def read(self) -> md.MuData:
        sage_result_df, sage_quant_df, sage_config = self._import_sage()
        mdata:md.MuData = self._sage2mdata(sage_result_df, sage_quant_df, sage_config)
        mdata = self._assign_protein_id_info(mdata=mdata)

        return mdata


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
        mdata: md.MuData = md.MuData({"feature": adata})
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
            pd.DataFrame(index=sage_result_df.index, columns=sage_quant_df.columns).T.astype("float")
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

        mdata = md.MuData({"feature": adata_psm, "peptide": adata_peptide})
        mdata = self._add_obs_tag(mdata=mdata, rename_dict=rename_dict)
        mdata.update_obs()

        return mdata
