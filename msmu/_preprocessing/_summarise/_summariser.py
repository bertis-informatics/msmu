import re
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from Bio import SeqIO


class Summariser:
    def __init__(self, adata: ad.AnnData, from_: str, to_: str) -> None:
        self._from: str = from_
        self._to: str = to_
        self._adata: ad.AnnData = adata.copy()
        self._obs: list[str] = adata.obs.index.tolist()

        self._agg_dict: dict = dict()
        self._rename_dict: dict[str, str] = dict()
        self._col_to_groupby: str = ""

    def summarise_data(self, data: pd.DataFrame, sum_method: str) -> pd.DataFrame:
        concated_agg_dict: dict = self._concat_agg_dict_w_obs(sum_method=sum_method)
        data = data.rename_axis(None)

        summarised_data: pd.DataFrame = data.groupby(
            self._col_to_groupby, observed=False
        ).agg(concated_agg_dict)

        summarised_data: pd.DataFrame = summarised_data.rename_axis(index=None)
        summarised_data: pd.DataFrame = summarised_data.rename(
            columns=self._rename_dict
        )

        return summarised_data

    def _concat_agg_dict_w_obs(self, sum_method: str) -> dict:
        concat_agg_dict: dict = self._agg_dict.copy()
        value_agg_dict: dict = {k: sum_method for k in self._obs}
        concat_agg_dict.update(value_agg_dict)

        return concat_agg_dict

    def data2adata(self, data: pd.DataFrame) -> ad.AnnData:
        var_cols: list[str] = [x for x in data.columns if x not in self._obs]
        var_data: pd.DataFrame = data[var_cols]
        arr_data: pd.DataFrame = data[self._obs].transpose()
        if arr_data.empty:
            var_data = var_data.astype("object")
            obs_data = pd.DataFrame(index=arr_data.index)
            arr_data = np.empty((len(obs_data), 0))
            adata: ad.AnnData = ad.AnnData(X=arr_data, var=var_data, obs=obs_data)
        else:
            adata: ad.AnnData = ad.AnnData(X=arr_data, var=var_data)
        adata.uns["level"] = self._to

        return adata

    def rank_(self, data: pd.DataFrame, rank_method: str) -> pd.DataFrame:
        if rank_method == "max_intensity":
            data.loc[:, "max_intensity"] = data[self._obs].sum(axis=1)
            data.loc[:, "rank"] = data.groupby(self._col_to_groupby)[
                "max_intensity"
            ].rank(ascending=False)
        else:
            raise ValueError(
                f"Unknown rank method: {rank_method}. Please choose from ['max_intensity']"
            )

        return data

    def filter_by_rank(self, data: pd.DataFrame, top_n: int) -> pd.DataFrame:
        data = data[data["rank"] <= top_n]

        return data


class PeptideSummariser(Summariser):
    def __init__(
        self,
        adata: ad.AnnData,
        peptide_col: str,
        protein_col: str,
    ) -> None:
        super().__init__(adata=adata, from_="feature", to_="peptide")

        self._col_to_groupby: str = peptide_col
        self._protein_col: str = protein_col

        self._agg_dict: dict[str, str] = {
            self._col_to_groupby: "first",
            self._protein_col: "first",
            "stripped_peptide": "first",
            # "modifications": "first",
            "total_psm": "first",
            "peptide": "count",
        }
        if "peptide_type" in self._adata.var.columns:
            self._agg_dict["peptide_type"] = "first"
        if "repr_protein" in self._adata.var.columns:
            self._agg_dict["repr_protein"] = "first"

        self._rename_dict: dict[str, str] = {"peptide": "num_used_psm"}

    def get_data(self) -> pd.DataFrame:
        adata = self._adata
        data: pd.DataFrame = adata.to_df().transpose()

        data[self._col_to_groupby] = adata.var[self._col_to_groupby].astype(str)
        data["peptide"] = adata.var["peptide"].astype(str)
        # data["modifications"] = adata.var["modifications"]
        data["stripped_peptide"] = adata.var["stripped_peptide"]
        data[self._protein_col] = adata.var[self._protein_col]
        if "peptide_type" in adata.var.columns:
            data["peptide_type"] = adata.var["peptide_type"]
        if "repr_protein" in adata.var.columns:
            data["repr_protein"] = adata.var["repr_protein"]

        for col in ["repr_protein", "peptide_type"]:
            if col in adata.var.columns:
                data[col] = adata.var[col]

        peptide_count: pd.DataFrame = (
            data[self._col_to_groupby].value_counts().to_frame("total_psm")
        )

        data: pd.DataFrame = data.merge(
            peptide_count, left_on=self._col_to_groupby, right_index=True
        )

        return data


class ProteinSummariser(Summariser):
    def __init__(self, adata, protein_col, from_) -> None:
        super().__init__(adata=adata, from_=from_, to_="protein")

        self._col_to_groupby: str = protein_col

        self._agg_dict: dict[str, str] = {
            "total_psm": "sum",
            "num_used_psm": "sum",
            "stripped_peptide": "nunique",
        }
        if "repr_protein" in self._adata.var.columns:
            self._agg_dict["repr_protein"] = "first"

        self._rename_dict: dict[str, str] = {"stripped_peptide": "num_peptides"}

    def get_data(self) -> pd.DataFrame:
        adata: ad.AnnData = self._adata
        data: pd.DataFrame = adata.to_df().transpose()

        data[self._col_to_groupby] = adata.var[self._col_to_groupby]
        data["stripped_peptide"] = adata.var["stripped_peptide"]

        for col in ["repr_protein", "peptide_type"]:
            if col in adata.var.columns:
                data[col] = adata.var[col]

        # make level_df(parent) and agg method dict for protein summarisation
        data["peptide"] = adata.var.index
        data["total_psm"] = adata.var["total_psm"]
        data["num_used_psm"] = adata.var["num_used_psm"]

        return data

    def filter_unique_peptides(self, data) -> pd.DataFrame:
        if "peptide_type" in data.columns:
            unique_data: pd.DataFrame = data[data["peptide_type"] == "unique"]
        else:
            unique_data: pd.DataFrame = data[
                len(data["num_used_psm"].str.split(";")) == 1
            ]

        return unique_data

    def filter_n_min_peptides(
        self, data: pd.DataFrame, min_n_peptides: int
    ) -> pd.DataFrame:
        data = data[data["num_peptides"] >= min_n_peptides]

        return data


class PtmSummariser(Summariser):
    def __init__(self, adata: ad.AnnData, protein_col: str) -> None:
        super().__init__(adata=adata, from_="peptide", to_="ptm")

        self._col_to_groupby: str = "protein_site"
        self._col_to_label: str = protein_col

        self._agg_dict: dict[str, str] = {
            "modified_protein": "first",
            "protein_group": "first",
            "peptide": "nunique",
            "num_used_psm": "sum",
        }
        if "repr_protein" in self._adata.var.columns:
            self._agg_dict["repr_protein"] = "first"

        self._rename_dict: dict[str, str] = {
            "peptide": "num_peptides",
            # "_prot_gr": "protein_group",
        }

    def get_data(self) -> pd.DataFrame:
        adata: ad.AnnData = self._adata
        data: pd.DataFrame = adata.to_df().transpose()

        data[self._col_to_label] = adata.var[self._col_to_label]
        data["stripped_peptide"] = adata.var["stripped_peptide"]

        for col in ["repr_protein", "peptide_type"]:
            if col in adata.var.columns:
                data[col] = adata.var[col]

        data["peptide"] = adata.var.index
        data["total_psm"] = adata.var["total_psm"]
        data["num_used_psm"] = adata.var["num_used_psm"]

        return data

    def label_ptm_site(
        self, data: pd.DataFrame, modification_mass: float, fasta_file: str | Path
    ) -> pd.DataFrame:
        """
        Label PTM site to each single protein and get data arranged by peptide - peptide site
        1. Filter data with only modified peptides with modi_identifier
        2. Get modified sites from peptide
        3. Label peptide site
        4. Explode data to single protein for labeling protein site
        5. Label protein site to each single protein
        6. Wrap up single protein to single protein group
        7. Group by modified peptide and its peptide site
        8. Merge data with peptide value indexed by peptide

        Args:
            data (pd.DataFrame): Peptide data from msmu mudata['peptide']
            modification_mass (float): Modification mass (as modification identifier to split peptide)
            fasta_file (str | Path): Fasta file
        output:
            ptm_data (pd.DataFrame): PTM data arranged by peptide - peptide site
        """
        modi_identifier: str = f"[+{modification_mass}]"

        # filter data with only modified peptides with modi_identifier
        modified_df: pd.DataFrame = self._get_modified_peptide_df(
            data=data, modi_identifier=modi_identifier
        ).copy()

        info_cols: list[str] = [x for x in modified_df.columns if x not in self._obs]
        ptm_info: pd.DataFrame = modified_df[info_cols].copy()
        ptm_info["peptide_site"] = ptm_info["peptide"].apply(
            lambda x: self._get_mod_sites(x, modi_identifier)
        )

        # label peptide site
        ptm_info["peptide_site"] = ptm_info["peptide_site"].apply(
            lambda x: self._label_peptide_site(x)
        )

        # explode data to single protein for label protein site
        ptm_info = self._explode_mod_site(ptm_info)
        ptm_info = self._explode_protein_groups(ptm_info)
        ptm_info = self._explode_protein_group(ptm_info)

        # label protein site to each single protein
        fasta_dict: dict = self._read_fasta_seq(file=fasta_file)
        ptm_info["protein_site"] = ptm_info.apply(
            lambda x: self._label_protein_site(
                protein=x._prots,
                peptide=x.stripped_peptide,
                pep_site=x.peptide_site,
                fasta_dict=fasta_dict,
            ),
            axis=1,
        )
        ptm_info = ptm_info.loc[ptm_info["protein_site"].str.len() > 0].copy()
        ptm_info["modified_protein"] = ptm_info["protein_site"].apply(
            lambda x: x.split("|")[0]
        )

        # wrap up single protein to single protein group
        ptm_info = self._implode_protein_group(ptm_info)

        # group by modified peptide and its peptide site
        ptm_info = self._implode_peptide_peptide_site(ptm_info)

        peptide_value: pd.DataFrame = modified_df[self._obs].copy()
        peptide_value["peptide"] = peptide_value.index
        ptm_data = pd.merge(ptm_info, peptide_value, how="left", on="peptide")

        return ptm_data

    def _get_modified_peptide_df(
        self, data: pd.DataFrame, modi_identifier: str
    ) -> pd.DataFrame:
        regex_modi_identifier: str = re.escape(modi_identifier)

        print(f"Total peptides: {len(data)}")
        print(f"Modi identifier: {modi_identifier}")
        data = data.loc[data["peptide"].str.contains(regex_modi_identifier)].copy()
        print(f"Modified peptides: {len(data)}")

        return data

    def _get_mod_sites(self, pep: str, modi_identifier) -> list:
        mod_sites: list = pep.split(modi_identifier)
        mod_sites: list = mod_sites[:-1]

        return mod_sites

    def _label_peptide_site(self, mod_sites: list) -> list:
        sites = list()
        site_pos: int = 0
        for mod in mod_sites:
            mod = "".join(filter(str.isalpha, mod))
            site_pos = site_pos + len(mod)
            site = f"{mod[-1]}{site_pos}"
            sites.append(site)

        return sites

    def _label_protein_site(
        self, protein: str, peptide: str, pep_site: str, fasta_dict: dict
    ) -> str:
        aa: str = pep_site[0]
        pos: int = int(pep_site[1:])
        prot_site: str = ""

        res: list = list()
        prot_split = self._get_uniprot(protein)

        if prot_split in fasta_dict.keys():
            refseq: str = fasta_dict[prot_split]
            for match in re.finditer(peptide, refseq):
                matched = f"{prot_split}|{aa}{pos + match.span()[0]}"
                res.append(matched)
            prot_site = "/".join(res)

        return prot_site

    def _explode_mod_site(self, pep_labed_data: pd.DataFrame) -> pd.DataFrame:
        pep_labed_data = pep_labed_data.explode("peptide_site", ignore_index=True)

        return pep_labed_data

    def _explode_protein_groups(self, pep_labed_data: pd.DataFrame) -> pd.DataFrame:
        pep_labed_data["_prot_gr"] = pep_labed_data["protein_group"]
        pep_labed_data["_prot_gr"] = pep_labed_data["_prot_gr"].str.split(";")
        exploded_data = pep_labed_data.explode("_prot_gr", ignore_index=True)

        return exploded_data

    def _explode_protein_group(self, data) -> pd.DataFrame:
        data["_prots"] = data["_prot_gr"]
        data["_prots"] = data["_prots"].str.split(",")
        exploded_data = data.explode("_prots", ignore_index=True)

        return exploded_data

    def _implode_protein_group(self, data) -> pd.DataFrame:
        data = (
            data.groupby(["peptide", "peptide_site", "_prot_gr"], as_index=False)
            .agg(
                {
                    "protein_site": ",".join,
                    "protein_group": "first",
                    "modified_protein": ",".join,
                    "stripped_peptide": "first",
                    "total_psm": "sum",
                    "num_used_psm": "sum",
                    "repr_protein": "first",
                }
            )
            .copy()
        )

        return data

    def _implode_peptide_peptide_site(self, data) -> pd.DataFrame:
        data = data.groupby(["peptide", "peptide_site"], as_index=False).agg(
            {
                "protein_site": ";".join,
                "protein_group": "first",
                "modified_protein": ";".join,
                "stripped_peptide": "first",
                "total_psm": "sum",
                "num_used_psm": "sum",
                "repr_protein": "first",
            }
        )
        print(data)

        return data

    def _read_fasta_seq(self, file: str | Path) -> dict[str, str]:
        result: dict[str, str] = dict()
        for record in SeqIO.parse(file, "fasta"):
            ref_uniprot: list[str] = record.id.split("|")[1]
            ref_seq: str = str(record.seq)
            if ref_uniprot in result:
                # print("skipping:", record.description)
                continue
            result[ref_uniprot] = ref_seq

        return result

    def _get_uniprot(self, protein: str) -> str:
        return protein