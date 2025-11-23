"""
Module for preparing plotting data from MuData objects.
"""

import mudata as md
import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import is_categorical_dtype  # type: ignore
from typing import TypedDict, cast

from ._utils import resolve_obs_column, BinInfo


class PlotData:
    def __init__(
        self,
        mdata: md.MuData,
        modality: str,
        **kwargs: str,
    ):
        """
        Prepares MuData observations, variables, and derived summaries for plotting.

        Parameters:
            mdata: MuData object containing observations and variables.
            modality: Modality key for accessing the appropriate AnnData object.
            **kwargs: Optional arguments including `obs_column` preference.
        """
        self.mdata = mdata
        self.modality = modality
        self.kwargs = kwargs
        self._default_obs_column = resolve_obs_column(self.mdata, kwargs.get("obs_column"))

    def get_adata(self) -> AnnData:
        """Returns the modality-specific AnnData object with proper typing."""
        return cast(AnnData, self.mdata[self.modality])

    def _get_data(self) -> pd.DataFrame:
        """
        Retrieves the expression/intensity DataFrame for the current modality.

        Returns:
            pd.DataFrame: Copy of the modality's data matrix as a DataFrame.
        """
        return self.get_adata().to_df().copy()

    def get_var(self) -> pd.DataFrame:
        """
        Retrieves the variable metadata for the current modality.

        Returns:
            pd.DataFrame: Copy of the modality's `var` table.
        """
        return self.mdata[self.modality].var.copy()

    def get_varm(self, column: str) -> pd.DataFrame:
        """
        Retrieves a varm column and merges it with `var` for plotting.

        Parameters:
            column: Name of the varm column to merge with `var`.

        Returns:
            Concatenated `var` and selected varm DataFrame.
        """
        var_df: pd.DataFrame = self.get_var()
        varm_df: pd.DataFrame = pd.DataFrame(self.mdata[self.modality].varm[column].copy())

        return pd.concat([var_df, varm_df], axis=1)

    def resolve_obs_column(self, obs_column: str | None = None) -> str:
        """
        Resolves and validates the observation column to use for grouping.

        Parameters:
            obs_column: Preferred observation column; falls back to default.

        Returns:
            elected observation column name.
        """
        if obs_column is None:
            return self._default_obs_column
        return resolve_obs_column(self.mdata, obs_column)

    def get_obs(self, obs_column: str) -> pd.DataFrame:
        """
        Retrieves observation metadata sorted and cast to categorical.

        Parameters:
            obs_column: Observation column used for ordering and grouping.

        Returns:
            Observation DataFrame with categorical ordering applied.
        """
        obs_column = self.resolve_obs_column(obs_column)
        obs_df = self.mdata.obs.copy()
        if "condition" in obs_df.columns:
            obs_df = obs_df.sort_values(["condition", obs_column])
        if not is_categorical_dtype(obs_df[obs_column]):
            obs_df[obs_column] = obs_df[obs_column].astype("category")
        obs_df[obs_column] = obs_df[obs_column].cat.remove_unused_categories()
        obs_df[obs_column] = obs_df[obs_column].cat.reorder_categories(obs_df[obs_column].values.tolist())

        return obs_df

    def prep_var_data(
        self,
        groupby: str,
        name: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares variable-level counts grouped by an observation column.

        Parameters:
            groupby: Observation column to group by.
            name: Variable column whose values define categories.
            obs_column: Observation column to align with variables.

        Returns:
            Aggregated counts per group and variable category.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
        orig_df = self._get_data()

        if (np.nansum(orig_df) == 0) or (groupby == "fraction"):
            prep_df = var_df.copy()
            if np.nansum(orig_df) == 0:
                print("No data available for the selected modality. Counting from var.")
            if groupby == "fraction":
                var_df["fraction"] = var_df["filename"]
                categories = pd.Categorical(pd.Index(var_df["fraction"].unique()).sort_values())

                if self.modality != "feature":
                    raise ValueError("groupby: 'fraction' only supports modality: 'feature'")
                if name == "id_count":
                    var_df["id_count"] = var_df["filename"]
            else:
                categories = obs_df[groupby].unique()

            if groupby not in var_df.columns:
                raise ValueError(f"Column '{groupby}' not found in var data.")

            prep_df = var_df[[groupby, name]].groupby(groupby, observed=True).value_counts().reset_index()
            prep_df[groupby] = pd.Categorical(prep_df[groupby], categories=categories)
            prep_df = prep_df.sort_values(groupby).reset_index(drop=True)
        else:
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

    def prep_var_bar(
        self,
        groupby: str,
        name: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares stacked bar data from variable annotations.

        Parameters:
            groupby: Observation column to group by.
            name: Variable column defining stacked categories.
            obs_column: Observation column to align with variables.

        Returns:
            Counts of variable categories per observation group.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
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

    def prep_var_box(
        self,
        groupby: str,
        var_column: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares variable values for box plot visualization.

        Parameters:
            groupby: Observation column to group by.
            var_column: Variable column containing numeric values.
            obs_column: Observation column to align with variables.

        Returns:
            Box-plot-ready DataFrame with grouping labels.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
        orig_df = self.get_adata().to_df()

        var_df = var_df[[var_column]]

        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        prep_df = melt_df.merge(var_df[[var_column]], left_on="_var", right_index=True)
        prep_df = prep_df[prep_df["_exists"] > 0]
        prep_df = prep_df.drop(["_var", "_exists"], axis=1)

        return prep_df

    def prep_var_simple_box(
        self,
        groupby: str,
        var_column: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares summary statistics for simplified box plots.

        Parameters:
            groupby: Observation column to group by.
            var_column: Variable column containing numeric values.
            obs_column: Observation column to align with variables.

        Returns:
            Descriptive statistics indexed by observation group.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
        orig_df = self.get_adata().to_df()

        var_df = var_df[[var_column]]

        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        prep_df = melt_df.merge(var_df[var_column], left_on="_var", right_index=True)
        prep_df = prep_df[prep_df["_exists"] > 0]
        prep_df = prep_df.drop(["_var", "_exists"], axis=1)

        prep_df = prep_df.groupby(groupby, observed=True).describe().droplevel(level=0, axis=1)
        prep_df.index = pd.CategoricalIndex(prep_df.index, categories=obs_df[groupby].unique())
        return prep_df

    def prep_var_hist(
        self,
        groupby: str,
        var_column: str,
        obs_column: str,
        bin_info: BinInfo,
    ) -> pd.DataFrame:
        """
        Prepares histogram-based counts for variable annotations.

        Parameters:
            groupby: Observation column to group by.
            var_column: Variable column containing numeric values.
            obs_column: Observation column to align with variables.
            bin_info: Precomputed bin edges, centers, and labels.

        Returns:
            Histogram counts and frequencies per observation group.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
        orig_df = self._get_data()
        n_bins = len(bin_info["labels"])
        var_df = var_df[[var_column]]

        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        prep_df = melt_df.merge(var_df[var_column], left_on="_var", right_index=True)
        prep_df = prep_df[prep_df["_exists"] > 0]
        prep_df = prep_df.drop(["_var", "_exists"], axis=1)

        prep_df["_bin_"] = pd.cut(
            prep_df[var_column],
            bins=bin_info["edges"],
            labels=bin_info["labels"],
            include_lowest=True,
        )

        grouped = prep_df.groupby([groupby, "_bin_"], observed=False).size().unstack(fill_value=0)
        grouped = grouped[grouped.sum(axis=1) > 0]
        grouped.index = pd.CategoricalIndex(grouped.index, categories=obs_df[groupby].unique())
        grouped = grouped.sort_index(axis=0)

        bin_counts = grouped.values.flatten()
        bin_freqs = bin_counts / prep_df.shape[0]
        bin_names = grouped.index.get_level_values(0).repeat(n_bins).tolist()

        # make dataframe
        prepped = pd.DataFrame(
            {
                "center": bin_info["centers"] * len(grouped),
                "label": bin_info["labels"] * len(grouped),
                "count": bin_counts,
                "frequency": bin_freqs,
                "name": bin_names,
            }
        )
        prepped["name"] = pd.Categorical(prepped["name"], categories=obs_df[groupby].unique())

        return prepped

    def prep_id_data(
        self,
        groupby: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Counts identified variables per observation group.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Counts per observation group with column `_count`.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
        orig_df = self._get_data()

        if np.nansum(orig_df) == 0:
            print("No data available for the selected modality. Counting from var.")
            if groupby not in var_df.columns:
                raise ValueError(f"Column '{groupby}' not found in var data.")
            prep_df = var_df[groupby].value_counts().reset_index()
        else:
            melt_df = orig_df.notna().groupby(obs_df[groupby], observed=True).any().T
            prep_df = melt_df.sum().reset_index()

        prep_df.columns = [groupby, "_count"]

        return prep_df

    def prep_intensity_data_hist(self, groupby: str, obs_column: str, bin_info: BinInfo) -> pd.DataFrame:
        """
        Calculates histogram bins for intensity distributions by group.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.
            bin_info: Precomputed bin metadata for binning.

        Returns:
            Histogram counts and frequencies per group and bin.
        """
        obs_df = self.get_obs(obs_column)
        orig_df = self._get_data().T
        n_bins = len(bin_info["labels"])

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        melt_df = melt_df.join(obs_df, on="_obs", how="left")

        melt_df["_bin_"] = pd.cut(
            melt_df["_value"],
            bins=bin_info["edges"],
            labels=bin_info["labels"],
            include_lowest=True,
        )

        grouped = melt_df.groupby([groupby, "_bin_"], observed=False).size().unstack(fill_value=0)
        grouped = grouped[grouped.sum(axis=1) > 0]
        grouped.index = pd.CategoricalIndex(grouped.index, categories=obs_df[groupby].unique())
        grouped = grouped.sort_index(axis=0)

        bin_counts = grouped.values.flatten()
        bin_freqs = bin_counts / melt_df.shape[0]
        bin_names = grouped.index.get_level_values(0).repeat(n_bins).tolist()

        # make dataframe
        prepped = pd.DataFrame(
            {
                "center": bin_info["centers"] * len(grouped),
                "label": bin_info["labels"] * len(grouped),
                "count": bin_counts,
                "frequency": bin_freqs,
                "name": bin_names,
            }
        )
        prepped["name"] = pd.Categorical(prepped["name"], categories=obs_df[groupby].unique())

        return prepped

    def get_bin_info(self, data: pd.Series, bins: int) -> BinInfo:
        """
        Computes histogram bin metadata for numeric intensity data.

        Parameters:
            data: Numeric data for binning.
            bins: Number of bins to divide the data into.

        Returns:
            Bin width, edges, centers, and labels.
        """
        values = np.asarray(data, dtype=float).flatten()
        if values.size == 0:
            raise ValueError("Cannot compute bin info for empty data.")

        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        data_range = max_value - min_value
        bin_width = data_range / bins if bins > 0 else 0
        bin_edges = [min_value + bin_width * i for i in range(bins + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]
        bin_labels = [f"{bin_edges[i]} - {bin_edges[i + 1]}" for i in range(bins)]

        return {
            "width": bin_width,
            "edges": bin_edges,
            "centers": bin_centers,
            "labels": bin_labels,
        }

    def prep_intensity_data(
        self,
        groupby: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares melted intensity values for violin/box plotting.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Long-form DataFrame with intensity values and groups.
        """
        obs_df = self.get_obs(obs_column)
        orig_df = self.get_adata().to_df().T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        join_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = join_df[[groupby, "_value"]]

        return prep_df

    def prep_intensity_data_box(
        self,
        groupby: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Aggregates intensity values into descriptive statistics by group.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Descriptive statistics indexed by the grouping column.
        """
        obs_df = self.get_obs(obs_column)
        orig_df = self.get_adata().to_df().T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        join_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = join_df[[groupby, "_value"]].groupby(groupby, observed=True).describe().droplevel(level=0, axis=1)
        prep_df.index = pd.CategoricalIndex(prep_df.index, categories=obs_df[groupby].unique())
        prep_df = prep_df.sort_index(axis=0)

        return prep_df

    def prep_missingness_data(
        self,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Computes cumulative missingness percentages across observations.

        Parameters:
            obs_column: Observation column used for ordering.

        Returns:
            Missingness ratios and counts ready for plotting.
        """
        obs = self.get_obs(obs_column)
        n_sample = obs.shape[0]

        # Prepare data
        orig_df = self.get_adata().to_df()
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

    def prep_pca_data(
        self,
        modality: str,
        groupby: str,
        pc_columns: list[str],
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares PCA coordinates joined with observation group labels.

        Parameters:
            modality: Modality key for accessing PCA embeddings.
            groupby: Observation column to group by.
            pc_columns: Names of PC columns to plot.
            obs_column: Observation column to align with variables.

        Returns:
            PCA coordinates with grouping metadata.
        """
        obs = self.get_obs(obs_column)

        # Prepare data
        orig_df = pd.DataFrame(self.mdata[modality].obsm["X_pca"][pc_columns])
        join_df = orig_df.join(obs, how="left")
        join_df[groupby] = pd.Categorical(join_df[groupby], categories=obs[groupby].unique())

        return join_df

    def prep_umap_data(
        self,
        modality: str,
        groupby: str,
        umap_columns: list[str],
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares UMAP coordinates joined with observation group labels.

        Parameters:
            modality: Modality key for accessing UMAP embeddings.
            groupby: Observation column to group by.
            umap_columns: Names of UMAP columns to plot.
            obs_column: Observation column to align with variables.

        Returns:
            UMAP coordinates with grouping metadata.
        """
        obs = self.get_obs(obs_column)

        # Prepare data
        orig_df = pd.DataFrame(self.mdata[modality].obsm["X_umap"][umap_columns])
        join_df = orig_df.join(obs, how="left")
        join_df[groupby] = pd.Categorical(join_df[groupby], categories=obs[groupby].unique())

        return join_df

    def prep_purity_data(
        self,
        groupby: str,
    ) -> pd.DataFrame:
        """
        Retrieves purity values optionally grouped by observation metadata.

        Parameters:
            groupby: Observation column to group by; None groups all together.

        Returns:
            Purity observations filtered to valid values.
        """
        data = self.get_var()

        if groupby is not None:
            data = data[[groupby, "purity"]]
        else:
            data = data[["purity"]]
            data["_idx_"] = "Purity"

        return data[data["purity"] >= 0]

    def prep_purity_data_hist(
        self,
        data: pd.DataFrame,
        bin_info: BinInfo,
        groupby: str | None = None,
    ) -> pd.DataFrame:
        """
        Prepares histogram bins for purity metrics.

        Parameters:
            data: Purity data containing a `purity` column.
            bin_info: Precomputed bin metadata for binning.
            groupby: Observation column to group by.

        Returns:
            Histogram counts/frequencies with labels per group.
        """
        df = data.copy()
        n_bins = len(bin_info["labels"])

        # Treat groupby
        df["_bin_"] = pd.cut(
            df["purity"],
            bins=bin_info["edges"],
            labels=bin_info["labels"],
            include_lowest=True,
        )
        if groupby is not None:
            grouped = df.groupby([groupby, "_bin_"], observed=False).size().unstack(fill_value=0)
            bin_counts = np.asarray(grouped.values.flatten(), dtype=float)
            bin_frequencies = bin_counts / float(df.shape[0])
            bin_names = grouped.index.get_level_values(0).repeat(n_bins).tolist()

            # make dataframe
            prepped = pd.DataFrame(
                {
                    "center": bin_info["centers"] * len(grouped),
                    "label": bin_info["labels"] * len(grouped),
                    "count": bin_counts,
                    "frequency": bin_frequencies,
                    "name": bin_names,
                }
            )
        else:
            bin_counts = np.asarray(df["_bin_"].value_counts(sort=False).values, dtype=float)
            bin_frequencies = bin_counts / float(df.shape[0])

            # make dataframe
            prepped = pd.DataFrame(
                {
                    "center": bin_info["centers"],
                    "label": bin_info["labels"],
                    "count": bin_counts,
                    "frequency": bin_frequencies,
                    "name": "Purity",
                }
            )
        return prepped

    def prep_purity_data_box(
        self,
        groupby: str,
    ) -> pd.DataFrame:
        """
        Builds descriptive statistics for purity values by group.

        Parameters:
            groupby: Observation column to group purity values.

        Returns:
            Descriptive statistics for purity per group.
        """
        # Prepare data
        orig_df = self.get_var()[[groupby, "purity"]]
        orig_df[["purity"]] = orig_df[["purity"]][orig_df[["purity"]] >= 0]

        prep_df = orig_df.groupby(groupby, observed=True).describe().droplevel(0, axis=1)

        return prep_df

    def prep_purity_data_vln(
        self,
        groupby: str,
    ) -> pd.DataFrame:
        """
        Prepares raw purity values for violin plotting.

        Parameters:
            groupby: Observation column to group purity values.

        Returns:
            Purity values with grouping labels.
        """
        # Prepare data
        orig_df = self.get_var()[[groupby, "purity"]]
        orig_df[["purity"]] = orig_df[["purity"]][orig_df[["purity"]] >= 0]

        prep_df = orig_df

        return prep_df

    def prep_peptide_length_data(
        self,
        groupby: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Derives peptide length distributions grouped by observations.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Descriptive statistics for peptide lengths per group.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
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

    def prep_peptide_length_data_vln(
        self,
        groupby: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares peptide length values for violin plots grouped by observations.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Peptide lengths with grouping labels for plotting.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
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

    def prep_missed_cleavage(
        self,
        groupby: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Aggregates missed cleavages counts grouped by observations.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Missed cleavage counts per observation group.
        """
        obs_df = self.get_obs(obs_column)
        var_df = self.get_var()
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

    def prep_upset_data(
        self,
        groupby: str,
        obs_column: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Builds combination and item counts for Upset plots.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Combination counts and item counts.
        """
        orig_df = self._get_data()
        obs_df = self.get_obs(obs_column)

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

    def prep_correlation_data(self, groupby: str, obs_column: str) -> pd.DataFrame:
        """
        Computes pairwise Pearson correlations between grouped median profiles.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Lower-triangular correlation matrix with NaNs above diagonal.
        """
        orig_df = self._get_data()
        obs_df = self.get_obs(obs_column)
        corrs_df = orig_df.groupby(obs_df[groupby], observed=True).median().T.corr(method="pearson")

        for x in range(corrs_df.shape[0]):
            for y in range(corrs_df.shape[1]):
                if x < y:
                    corrs_df.iloc[x, y] = np.nan

        corrs_df = corrs_df.sort_index(axis=0).sort_index(axis=1)

        return corrs_df

    def prep_purity_metrics_data(
        self,
    ) -> pd.DataFrame:
        """
        Summarizes purity pass/fail metrics and ratios by filename.

        Returns:
            Purity classification counts and ratios per filename.
        """
        varm_df = self.get_varm("filter")

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
