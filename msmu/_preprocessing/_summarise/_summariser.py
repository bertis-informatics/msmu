import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import re
from pathlib import Path
from Bio import SeqIO


class Summariser:
    def __init__(self, adata: ad.AnnData, from_: str, to_: str) -> None:
        self._agg_dict: dict = dict()
        self._from: str = from_
        self._to: str = to_
        self._adata: ad.AnnData = adata.copy()
        self._obs: list[str] = adata.obs.index.tolist()

    def summarise_data(self, data:pd.DataFrame, sum_method:str) -> pd.DataFrame:
        concated_agg_dict:dict = self._concat_agg_dict_w_obs(sum_method=sum_method)

        summarised_data:pd.DataFrame = data.groupby(self._col_to_groupby, observed=False).agg(concated_agg_dict)

        summarised_data:pd.DataFrame = summarised_data.rename_axis(index=None)
        summarised_data:pd.DataFrame = summarised_data.rename(columns=self._rename_dict)

        return summarised_data

    def _concat_agg_dict_w_obs(self, sum_method:str) -> dict:
        concat_agg_dict:dict = self._agg_dict.copy()
        value_agg_dict:dict = {k: sum_method for k in self._obs}
        concat_agg_dict.update(value_agg_dict)

        return concat_agg_dict

    def data2adata(self, data:pd.DataFrame) -> ad.AnnData:
        var_cols:list[str] = [x for x in data.columns if x not in self._obs]
        var_data:pd.DataFrame = data[var_cols]
        arr_data:pd.DataFrame = data[self._obs].transpose()

        adata:ad.AnnData = ad.AnnData(X=arr_data, var=var_data)
        adata.uns["level"] = self._to

        return adata


class PeptideSummariser(Summariser):
    def __init__(self, adata:ad.AnnData, peptide_col:str, protein_col:str, from_:str) -> None:
        super().__init__(adata=adata, from_=from_, to_="peptide")

        self._col_to_groupby: str = peptide_col
        self._protein_col: str = protein_col

        self._agg_dict: dict[str, str] = {
            self._col_to_groupby: "first",
            self._protein_col: "first",
            "stripped_peptide": "first",
            "modifications": "first",
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

        data["peptide"] = adata.var["peptide"].astype(str)
        data["modifications"] = adata.var["modifications"]
        data["stripped_peptide"] = adata.var["stripped_peptide"]
        data[self._protein_col] = adata.var[self._protein_col]
        if "peptide_type" in adata.var.columns:
            data["peptide_type"] = adata.var["peptide_type"]
        if "repr_protein" in adata.var.columns:
            data["repr_protein"] = adata.var["repr_protein"]

        for col in ["repr_protein", "peptide_type"]:
            if col in adata.var.columns:
                data[col] = adata.var[col]

        peptide_count: pd.DataFrame = data[self._col_to_groupby].value_counts().to_frame("total_psm")

        data: pd.DataFrame = data.merge(peptide_count, left_on=self._col_to_groupby, right_index=True)

        return data

    def rank_psm(self, data: pd.DataFrame, rank_method: str) -> pd.DataFrame:
        if rank_method == "max_intensity":
            data.loc[:, "max_intensity"] = data[self._obs].max(axis=1)
            data.loc[:, "rank"] = data.groupby(self._col_to_groupby)["max_intensity"].rank(ascending=False)
        else:
            raise ValueError(f"Unknown rank method: {rank_method}. Please choose from ['max_intensity']")

        return data

    def filter_by_rank(self, data: pd.DataFrame, top_n: int) -> pd.DataFrame:
        data = data[data["rank"] <= top_n]

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
        adata:ad.AnnData = self._adata
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
            unique_data: pd.DataFrame = data[len(data["num_used_psm"].str.split(";")) == 1]

        return unique_data

    def filter_n_min_peptides(self, data: pd.DataFrame, min_n_peptides: int) -> pd.DataFrame:
        data = data[data["num_peptides"] >= min_n_peptides]

        return data


class PtmSummariser(Summariser):
    def __init__(self, mdata) -> None:
        super().__init__(mdata=mdata)
        self.agg_dict: dict[str, str|callable] = {
            "protein_group": "first",
            "stripped_peptide": "first",
        }

        self.data: pd.DataFrame = self.mdata.mod["peptide"].var
        self.arr = self.mdata.mod["peptide"].X.T

    def label_ptm_site(self, modification_mass: float, fasta_file: str | Path) -> pd.DataFrame:
        fasta_dict:dict = self._read_fasta_seq(file=fasta_file)

        mod_identifier = f"[+{modification_mass}]"

        ptm_list:list = []
        for idx, row in self.data.iterrows():
            pep:str = idx
            prot:list = row["protein_group"].split(";")

            mod_sites:list = pep.split(mod_identifier)
            mod_sites:list = mod_sites[:-1]

            sites:dict = dict()
            site_pos:int = 0
            for mod in mod_sites:
                mod = "".join(filter(str.isalpha, mod))
                site_pos = site_pos + len(mod)
                sites[f"{mod[-1]}{site_pos}"] = []

            if len(sites) < 1:
                continue

            stripped_peptide = "".join(filter(str.isalpha, pep))
            for site, ls in sites.items():
                aa:str = site[0]
                pos:int = int(site[1:])
                group:list = []

                for p in prot:
                    res:list = []
                    for pr in self._split_protein_group(p):
                        if pr not in fasta_dict:
                            continue

                        refseq:str = fasta_dict[pr]
                        for match in re.finditer(stripped_peptide, refseq):
                            res.append(f"{pr}|{aa}{pos + match.span()[0]}")

                    res:str = ";".join(res)
                    if res:
                        ls.append(res)
                        group.append(p)

                assert len(ls) == len(group), "Length does not match!"

                ls:str = ",".join(ls)
                group:str = ",".join(group)

                if ls:
                    ptm_list.append(dict(Peptide=pep, Peptide_site=site, Phosphosite=ls, Phosphoprotein=group))

        ptm_df:pd.DataFrame = pd.DataFrame(ptm_list)

        return ptm_df

    def _read_fasta_seq(self, file: Path) -> dict[str, str]:
        result:dict[str, str] = dict()
        for record in SeqIO.parse(file, "fasta"):
            ref_uniprot:list[str] = record.id.split("|")[1]
            ref_seq:str= str(record.seq)
            if ref_uniprot in result:
                print("skipping:", record.description)
                continue
            result[ref_uniprot] = ref_seq

        return result

    def _split_protein_group(self, protein_group: str) -> list[str]:
        result:list[str] = protein_group.split(";")
        for index, prot in enumerate(result):
            result[index] = prot.split("|")[1]
        return result

    def summarise(self, ptm_data: pd.DataFrame, arr: np.ndarray, sum_method: str) -> pd.DataFrame:
        pass

    def _summarise_arr(self, arr: np.ndarray, sum_method: str) -> np.ndarray:
        pass