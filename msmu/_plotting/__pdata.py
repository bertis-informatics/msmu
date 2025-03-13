import numpy as np
import pandas as pd
import mudata as md

DEFAULT_COLUMN = "_obs_"


class PlotData:
    def __init__(
        self,
        mdata: md.MuData,
        mods: list[str],
        **kwargs,
    ):
        self.mdata = mdata
        self.mods = mods
        self.kwargs = kwargs

    def _get_data(self):
        return pd.concat([self.mdata[mod].to_df() for mod in self.mods]).copy()

    def _get_var(self):
        return pd.concat([self.mdata[mod].var for mod in self.mods]).copy()

    def _get_obs(self):
        obs_df = self.mdata.obs.copy()
        obs_df[DEFAULT_COLUMN] = obs_df.index
        return obs_df

    def _prep_charge_data(
        self,
        groupby: str,
        name: str,
    ) -> pd.DataFrame:

        obs_df = self._get_obs()
        var_df = self._get_var()
        orig_df = self._get_data()

        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        prep_df = melt_df.merge(var_df[[name]], left_on="_var", right_index=True)
        prep_df = prep_df[prep_df["_exists"] > 0]
        prep_df = prep_df.drop(["_var", "_exists"], axis=1)
        prep_df = prep_df.groupby(groupby, observed=True).value_counts().reset_index()

        return prep_df

    def _prep_id_data(
        self,
        groupby: str,
    ):
        obs_df = self._get_obs()
        orig_df = pd.DataFrame(self._get_data().T.count(), columns=["count"]).T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_count").dropna()
        melt_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = pd.DataFrame(melt_df.groupby(groupby, observed=True)["_count"].mean(), columns=["_count"]).T
        prep_df = prep_df.melt(var_name=groupby, value_name="_count").dropna()

        return prep_df

    def _prep_intensity_data_hist(
        self,
        groupby: str,
        bins: int,
    ) -> pd.DataFrame:
        obs_df = self._get_obs()
        orig_df = self._get_data().T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        melt_df = melt_df.join(obs_df, on="_obs", how="left")

        # self.bin_info = self._get_bin_info(melt_df["_value"], bins)

        melt_df["_bin_"] = pd.cut(
            melt_df["_value"],
            bins=self.bin_info["edges"],
            labels=self.bin_info["labels"],
            include_lowest=True,
        )

        grouped = melt_df.groupby([groupby, "_bin_"], observed=False).size().unstack(fill_value=0)
        grouped = grouped[grouped.sum(axis=1) > 0]

        bin_counts = grouped.values.flatten()
        bin_freqs = bin_counts / melt_df.shape[0]
        bin_names = grouped.index.get_level_values(0).repeat(bins).tolist()

        # make dataframe
        prepped = pd.DataFrame(
            {
                "center": self.bin_info["centers"] * len(grouped),
                "label": self.bin_info["labels"] * len(grouped),
                "count": bin_counts,
                "frequency": bin_freqs,
                "name": bin_names,
            }
        )

        return prepped

    def _get_bin_info(self, data: pd.DataFrame, bins: int) -> dict:
        # get bin data
        min_value = np.min(data)
        max_value = np.max(data)
        data_range = max_value - min_value
        bin_width = data_range / bins
        bin_edges = [min_value + bin_width * i for i in range(bins + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]
        bin_labels = [f"{bin_edges[i]} - {bin_edges[i + 1]}" for i in range(bins)]

        self.bin_info = {
            "width": bin_width,
            "edges": bin_edges,
            "centers": bin_centers,
            "labels": bin_labels,
        }

        return self.bin_info

    def _prep_intensity_data_box(
        self,
        groupby: str,
    ) -> pd.DataFrame:
        obs_df = self._get_obs()
        orig_df = pd.concat([self.mdata[mod].to_df() for mod in self.mods]).T
        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        join_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = join_df[[groupby, "_value"]].groupby(groupby, observed=True).describe().droplevel(level=0, axis=1)

        return prep_df

    def _prep_missingness_data(
        self,
    ) -> pd.DataFrame:
        obs = self._get_obs()
        n_sample = obs.shape[0]

        # Prepare data
        orig_df = pd.concat([self.mdata[mod].to_df() for mod in self.mods])
        sum_list = orig_df.isna().sum(axis=0)

        count_list = sum_list.value_counts().sort_index().cumsum()
        count_list[np.int64(0)] = np.int64(0)
        count_list[n_sample] = np.int64(orig_df.shape[1])
        count_list = count_list.sort_index()

        prep_df = pd.DataFrame(count_list).reset_index(names="missingness")
        prep_df["ratio"] = prep_df["count"] / np.max(prep_df["count"]) * 100
        prep_df["missingness"] = prep_df["missingness"] / n_sample * 100
        prep_df["name"] = "Missingness"

        return prep_df

    def _prep_pca_data(
        self,
        modality: str,
        pc_columns: list[str],
    ) -> pd.DataFrame:
        obs = self._get_obs()

        # Prepare data
        orig_df = self.mdata[modality].obsm["X_pca"][pc_columns].reset_index(names="_obs")
        join_df = orig_df.join(obs, on="_obs", how="left")

        return join_df

    def _prep_umap_data(
        self,
        modality: str,
        umap_columns: list[str],
    ) -> pd.DataFrame:
        obs = self._get_obs()

        # Prepare data
        orig_df = self.mdata[modality].obsm["X_umap"][umap_columns].reset_index(names="_obs")
        join_df = orig_df.join(obs, on="_obs", how="left")

        return join_df

    def _prep_purity_data(
        self,
        groupby: str,
    ):
        data = self._get_var()

        if groupby is not None:
            data = data[[groupby, "purity"]]
        else:
            data = data[["purity"]]
            data["_idx_"] = "Purity"

        self.X = data[data["purity"] >= 0]

        return self.X

    def _prep_purity_data_hist(
        self,
        groupby: str = None,
        bins: int = 50,
    ):
        data = self.X

        # Treat groupby
        data["_bin_"] = pd.cut(
            data["purity"],
            bins=self.bin_info["edges"],
            labels=self.bin_info["labels"],
            include_lowest=True,
        )
        if groupby is not None:
            grouped = data.groupby([groupby, "_bin_"], observed=False).size().unstack(fill_value=0)
            bin_counts = grouped.values.flatten()
            bin_frequencies = bin_counts / data.shape[0]
            bin_names = grouped.index.get_level_values(0).repeat(bins).tolist()

            # make dataframe
            prepped = pd.DataFrame(
                {
                    "center": self.bin_info["centers"] * len(grouped),
                    "label": self.bin_info["labels"] * len(grouped),
                    "count": bin_counts,
                    "frequency": bin_frequencies,
                    "name": bin_names,
                }
            )
        else:
            bin_counts = data["_bin_"].value_counts(sort=False).values
            bin_frequencies = bin_counts / data.shape[0]

            # make dataframe
            prepped = pd.DataFrame(
                {
                    "center": self.bin_info["centers"],
                    "label": self.bin_info["labels"],
                    "count": bin_counts,
                    "frequency": bin_frequencies,
                    "name": "Purity",
                }
            )
        return prepped

    def _prep_purity_data_box(
        self,
        groupby: str,
    ) -> pd.DataFrame:
        # Prepare data
        orig_df = self._get_var()[[groupby, "purity"]]
        orig_df[["purity"]] = orig_df[["purity"]][orig_df[["purity"]] >= 0]

        prep_df = orig_df.groupby(groupby, observed=True).describe().droplevel(0, axis=1)

        return prep_df

    def _prep_peptide_length_data(
        self,
        groupby: str,
    ):
        obs_df = self._get_obs()
        var_df = self._get_var()
        orig_df = self._get_data()

        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        var_df["peptide_length"] = var_df["stripped_peptide"].str.len()
        prep_df = melt_df.merge(var_df[["peptide_length"]], left_on="_var", right_index=True)
        prep_df = prep_df[prep_df["_exists"] > 0]
        prep_df = prep_df.drop(["_var", "_exists"], axis=1)
        prep_df = prep_df.groupby(groupby, observed=True).value_counts().reset_index()

        return prep_df

    def _prep_missed_cleavage(
        self,
        groupby: str,
    ):
        obs_df = self._get_obs()
        var_df = self._get_var()
        orig_df = self._get_data()

        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        var_df["missed_cleavage"] = var_df["missed_cleavages"]
        prep_df = melt_df.merge(var_df[["missed_cleavage"]], left_on="_var", right_index=True)
        prep_df = prep_df[prep_df["_exists"] > 0]
        prep_df = prep_df.drop(["_var", "_exists"], axis=1)
        prep_df = prep_df.groupby(groupby, observed=True).value_counts().reset_index()

        return prep_df

    def _prep_upset_data(
        self,
    ):
        orig_df = self._get_data().T

        # Get the binary representation of the sets
        orig_df[orig_df.notna()] = 1
        orig_df[orig_df.isna()] = 0
        orig_df = orig_df.astype(int)
        df_binary = orig_df.apply(lambda row: "".join(row.astype(str)), axis=1)

        combination_counts = df_binary.sort_values(ascending=False).value_counts(sort=False).reset_index()
        combination_counts.columns = ["combination", "count"]
        item_counts = orig_df.sum()

        return combination_counts, item_counts
