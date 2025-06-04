import numpy as np
import pandas as pd
import mudata as md
import itertools

from ._utils import DEFAULT_COLUMN


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

    def _get_varm(self, column: str):
        var_df = self._get_var()
        varm_df = pd.concat([self.mdata[mod].varm[column] for mod in self.mods]).copy()
        return pd.concat([var_df, varm_df], axis=1)

    def _get_obs(self, obs_column: str = DEFAULT_COLUMN):
        obs_df = self.mdata.obs.copy()
        obs_df = obs_df.sort_values(["condition", obs_column])
        obs_df[obs_column] = obs_df[obs_column].cat.remove_unused_categories()
        obs_df[obs_column] = obs_df[obs_column].cat.reorder_categories(obs_df[obs_column].values.tolist())

        return obs_df

    def _prep_var_data(
        self,
        groupby: str,
        name: str,
        obs_column: str = DEFAULT_COLUMN,
    ) -> pd.DataFrame:
        obs_df = self._get_obs(obs_column)
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
        prep_df[groupby] = prep_df[groupby].values.tolist()

        prep_df[groupby] = pd.Categorical(prep_df[groupby], categories=obs_df[groupby].unique())
        prep_df = prep_df.sort_values(groupby)

        return prep_df

    def _prep_id_data(
        self,
        groupby: str,
        obs_column: str = DEFAULT_COLUMN,
    ):
        obs_df = self._get_obs(obs_column)
        orig_df = pd.DataFrame(self._get_data().T.count(), columns=["count"]).T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_count").dropna()
        melt_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = pd.DataFrame(melt_df.groupby(groupby, observed=True)["_count"].mean(), columns=["_count"]).T
        prep_df = prep_df.melt(var_name=groupby, value_name="_count").dropna()
        prep_df[groupby] = pd.Categorical(prep_df[groupby], categories=obs_df[groupby].unique())
        prep_df = prep_df.sort_index(axis=0)

        return prep_df

    def _prep_id_fraction_data(self, groupby: str) -> pd.DataFrame:
        return pd.DataFrame(self._get_var()[groupby].value_counts(sort=False)).reset_index()

    def _prep_intensity_data_hist(
        self,
        groupby: str,
        bins: int,
        obs_column: str = DEFAULT_COLUMN,
    ) -> pd.DataFrame:
        obs_df = self._get_obs(obs_column)
        orig_df = self._get_data().T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        melt_df = melt_df.join(obs_df, on="_obs", how="left")

        melt_df["_bin_"] = pd.cut(
            melt_df["_value"],
            bins=self.bin_info["edges"],
            labels=self.bin_info["labels"],
            include_lowest=True,
        )

        grouped = melt_df.groupby([groupby, "_bin_"], observed=False).size().unstack(fill_value=0)
        grouped = grouped[grouped.sum(axis=1) > 0]
        grouped.index = pd.CategoricalIndex(grouped.index, categories=obs_df[groupby].unique())
        grouped = grouped.sort_index(axis=0)

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
        prepped["name"] = pd.Categorical(prepped["name"], categories=obs_df[groupby].unique())

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

    def _prep_intensity_data(
        self,
        groupby: str,
        obs_column: str = DEFAULT_COLUMN,
    ) -> pd.DataFrame:
        obs_df = self._get_obs(obs_column)
        orig_df = pd.concat([self.mdata[mod].to_df() for mod in self.mods]).T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        join_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = join_df[[groupby, "_value"]]

        return prep_df

    def _prep_intensity_data_box(
        self,
        groupby: str,
        obs_column: str = DEFAULT_COLUMN,
    ) -> pd.DataFrame:
        obs_df = self._get_obs(obs_column)
        orig_df = pd.concat([self.mdata[mod].to_df() for mod in self.mods]).T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        join_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = join_df[[groupby, "_value"]].groupby(groupby, observed=True).describe().droplevel(level=0, axis=1)
        prep_df.index = pd.CategoricalIndex(prep_df.index, categories=obs_df[groupby].unique())
        prep_df = prep_df.sort_index(axis=0)

        return prep_df

    def _prep_missingness_data(
        self,
        obs_column: str = DEFAULT_COLUMN,
    ) -> pd.DataFrame:
        obs = self._get_obs(obs_column)
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
        groupby: str,
        pc_columns: list[str],
        obs_column: str = DEFAULT_COLUMN,
    ) -> pd.DataFrame:
        obs = self._get_obs(obs_column)

        # Prepare data
        orig_df = self.mdata[modality].obsm["X_pca"][pc_columns]
        join_df = orig_df.join(obs, how="left")
        join_df[groupby] = pd.Categorical(join_df[groupby], categories=obs[groupby].unique())

        return join_df

    def _prep_umap_data(
        self,
        modality: str,
        groupby: str,
        umap_columns: list[str],
        obs_column: str = DEFAULT_COLUMN,
    ) -> pd.DataFrame:
        obs = self._get_obs(obs_column)

        # Prepare data
        orig_df = self.mdata[modality].obsm["X_umap"][umap_columns]
        join_df = orig_df.join(obs, how="left")
        join_df[groupby] = pd.Categorical(join_df[groupby], categories=obs[groupby].unique())

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

    def _prep_purity_data_vln(
        self,
        groupby: str,
    ) -> pd.DataFrame:
        # Prepare data
        orig_df = self._get_var()[[groupby, "purity"]]
        orig_df[["purity"]] = orig_df[["purity"]][orig_df[["purity"]] >= 0]

        prep_df = orig_df

        return prep_df

    def _prep_peptide_length_data(
        self,
        groupby: str,
        obs_column: str = DEFAULT_COLUMN,
    ):
        obs_df = self._get_obs(obs_column)
        var_df = self._get_var()
        orig_df = self._get_data()
        var_df["peptide_length"] = var_df["stripped_peptide"].str.len()

        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        prep_df = melt_df.merge(var_df[["peptide_length"]], left_on="_var", right_index=True)
        prep_df = prep_df[prep_df["_exists"] > 0]
        prep_df = prep_df.drop(["_var", "_exists"], axis=1)
        prep_df = prep_df.groupby(groupby, observed=True).describe().droplevel(0, axis=1)
        prep_df.index = pd.CategoricalIndex(prep_df.index, categories=obs_df[groupby].unique())
        prep_df = prep_df.sort_index(axis=0)

        return prep_df

    def _prep_peptide_length_data_vln(
        self,
        groupby: str,
        obs_column: str = DEFAULT_COLUMN,
    ):
        obs_df = self._get_obs(obs_column)
        var_df = self._get_var()
        orig_df = self._get_data()
        var_df["peptide_length"] = var_df["stripped_peptide"].str.len()

        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        prep_df = melt_df.merge(var_df[["peptide_length"]], left_on="_var", right_index=True)
        prep_df = prep_df[prep_df["_exists"] > 0]
        prep_df = prep_df.drop(["_var", "_exists"], axis=1)

        return prep_df

    def _prep_missed_cleavage(
        self,
        groupby: str,
        obs_column: str = DEFAULT_COLUMN,
    ):
        obs_df = self._get_obs(obs_column)
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
        prep_df[groupby] = pd.Categorical(prep_df[groupby], categories=obs_df[groupby].unique())
        prep_df = prep_df.sort_index(axis=0)

        return prep_df

    def _prep_upset_data(
        self,
        groupby: str = DEFAULT_COLUMN,
        obs_column: str = DEFAULT_COLUMN,
    ):
        orig_df = self._get_data()
        obs_df = self._get_obs(obs_column)

        orig_df.index = pd.CategoricalIndex(orig_df.index, categories=obs_df.index)
        orig_df = orig_df.sort_index(axis=0)

        # Get the binary representation of the sets
        orig_df = orig_df.groupby(obs_df[groupby], observed=True).any()
        orig_df = orig_df.astype(int)
        df_binary = orig_df.apply(lambda row: "".join(row.astype(str)), axis=0)

        combination_counts = df_binary.sort_values(ascending=False).value_counts(sort=False).reset_index()
        combination_counts.columns = ["combination", "count"]
        combination_counts = combination_counts.sort_values(by="count", ascending=False)
        item_counts = orig_df.sum(axis=1)

        return combination_counts, item_counts

    def _prep_correlation_data(self, groupby: str, obs_column: str = DEFAULT_COLUMN):
        orig_df = self._get_data()
        obs_df = self._get_obs(obs_column)
        corrs_df = orig_df.groupby(obs_df[groupby], observed=True).median().T.corr(method="pearson")

        for x in range(corrs_df.shape[0]):
            for y in range(corrs_df.shape[1]):
                if x < y:
                    corrs_df.iloc[x, y] = np.nan

        corrs_df = corrs_df.sort_index(axis=0).sort_index(axis=1)

        return corrs_df

    def _prep_purity_metrics_data(
        self,
    ):
        varm_df = self._get_varm("filter")

        # Define conditions and choices for purity_result
        conditions = [
            varm_df["filter_purity"] == True,
            (varm_df["filter_purity"] == False) & (varm_df["purity"] >= 0),
            varm_df["purity"] == -1,
            varm_df["purity"] == -2,
        ]
        choices = [
            "High purity",
            "Low purity",
            "No isotope peak",
            "No isolation peak",
        ]
        varm_df["purity_metrics"] = np.select(condlist=conditions, choicelist=choices, default="Unknown")

        df = varm_df.groupby(["purity_metrics", "filename"], observed=True).size().reset_index(name="count")
        df["purity_metrics"] = pd.Categorical(df["purity_metrics"], categories=choices)
        df["ratio"] = df["count"] / df.groupby("filename", observed=False)["count"].transform("sum") * 100
        df = df.sort_values(["filename", "purity_metrics"]).reset_index(drop=True)

        return df
