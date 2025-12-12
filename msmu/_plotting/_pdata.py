"""
Module for preparing plotting data from MuData objects.
"""

from mudata import MuData
import numpy as np
import pandas as pd

from ._utils import resolve_obs_column, BinInfo
from .._utils.get import get_adata


class PlotData:
    def __init__(
        self,
        mdata: MuData,
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

    def _get_data(self) -> pd.DataFrame:
        """
        Retrieves the expression/intensity DataFrame for the current modality.

        Returns:
            Copy of the modality's data matrix as a DataFrame.
        """
        return get_adata(self.mdata, self.modality).to_df().copy()

    def _get_var(self) -> pd.DataFrame:
        """
        Retrieves the variable metadata for the current modality.

        Returns:
            Copy of the modality's `var` table.
        """
        return self.mdata[self.modality].var.copy()

    def _get_varm(self, column: str) -> pd.DataFrame:
        """
        Retrieves a varm column and merges it with `var` for plotting.

        Parameters:
            column: Name of the varm column to merge with `var`.

        Returns:
            Concatenated `var` and selected varm DataFrame.
        """
        var_df: pd.DataFrame = self._get_var()
        varm_df: pd.DataFrame = pd.DataFrame(self.mdata[self.modality].varm[column].copy())

        return pd.concat([var_df, varm_df], axis=1)

    def _get_obs(self, obs_column: str, groupby: str = "") -> pd.DataFrame:
        """
        Retrieves observation metadata sorted and cast to categorical.

        Parameters:
            obs_column: Observation column used for ordering and grouping.

        Returns:
            Observation DataFrame with categorical ordering applied.
        """
        obs_column = resolve_obs_column(self.mdata, obs_column)
        obs_df = self.mdata.obs.copy()

        if not isinstance(obs_df[obs_column].dtype, pd.CategoricalDtype):
            obs_df[obs_column] = pd.Categorical(obs_df[obs_column], categories=obs_df[obs_column].unique())

        obs_df[obs_column] = obs_df[obs_column].cat.remove_unused_categories()
        obs_df[obs_column] = obs_df[obs_column].cat.reorder_categories(obs_df[obs_column].values.tolist())

        if groupby and groupby != obs_column:
            if not isinstance(obs_df[groupby].dtype, pd.CategoricalDtype):
                obs_df[groupby] = pd.Categorical(obs_df[groupby], categories=obs_df[groupby].unique())

            obs_df[groupby] = obs_df[groupby].cat.remove_unused_categories()

        return obs_df

    def _get_bin_info(self, data: pd.DataFrame, bins: int) -> BinInfo:
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
        obs_df = self._get_obs(obs_column, groupby=groupby)
        var_df = self._get_var()
        orig_df = self._get_data()

        if (np.nansum(orig_df) == 0) or (groupby == "fraction"):
            prep_df = var_df.copy()
            if np.nansum(orig_df) == 0:
                print("No data available for the selected modality. Counting from var.")
            if groupby == "fraction":
                var_df["fraction"] = var_df["filename"]
                categories = pd.Categorical(pd.Index(var_df["fraction"].unique()).sort_values())

                if self.modality != "psm":
                    raise ValueError("groupby: 'fraction' only supports modality: 'psm'")
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
        var_column: str,
        obs_column: str,
    ) -> pd.DataFrame:
        """
        Prepares stacked bar data from variable annotations.

        Parameters:
            groupby: Observation column to group by.
            var_column: Variable column defining stacked categories.
            obs_column: Observation column to align with variables.

        Returns:
            Counts of variable categories per observation group.
        """
        obs_df = self._get_obs(obs_column, groupby=groupby)
        var_df = self._get_var()
        orig_df = self._get_data()

        if np.nansum(orig_df) == 0:
            print("No data available for the selected modality. Counting from var.")
            prep_df = var_df.copy()
            if groupby not in var_df.columns:
                raise ValueError(f"Column '{groupby}' not found in var data.")

            categories = var_df[groupby].unique()
            prep_df = var_df[[groupby, var_column]].groupby(groupby, observed=True).value_counts().reset_index()
            prep_df[groupby] = pd.Categorical(prep_df[groupby], categories=categories)
            prep_df = prep_df.sort_values(groupby).reset_index(drop=True)
        else:
            merged_df = orig_df.notna().join(obs_df[groupby], how="left")
            merged_df = merged_df.groupby(groupby, observed=True).any()

            melt_df = merged_df.stack().reset_index()
            melt_df.columns = [groupby, "_var", "_exists"]

            prep_df = melt_df.merge(var_df[[var_column]], left_on="_var", right_index=True)
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
        obs_df = self._get_obs(obs_column, groupby=groupby)
        var_df = self._get_var()
        orig_df = get_adata(self.mdata, self.modality).to_df()

        if np.nansum(orig_df) == 0:
            print("No data available for the selected modality. Counting from var.")
            prep_df = var_df.copy()
            if groupby not in var_df.columns:
                raise ValueError(f"Column '{groupby}' not found in var data.")

            prep_df = var_df[[groupby, var_column]]
        else:
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
        obs_df = self._get_obs(obs_column, groupby=groupby)
        var_df = self._get_var()
        orig_df = get_adata(self.mdata, self.modality).to_df()

        if np.nansum(orig_df) == 0:
            print("No data available for the selected modality. Counting from var.")
            prep_df = var_df.copy()
            if groupby not in var_df.columns:
                raise ValueError(f"Column '{groupby}' not found in var data.")

            prep_df = var_df[[groupby, var_column]]
            prep_df = prep_df.groupby(groupby, observed=True).describe().droplevel(level=0, axis=1)
            prep_df.index = pd.CategoricalIndex(prep_df.index, categories=obs_df[groupby].unique())
        else:
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
        obs_df = self._get_obs(obs_column, groupby=groupby)
        var_df = self._get_var()
        orig_df = self._get_data()
        n_bins = len(bin_info["labels"])

        if np.nansum(orig_df) == 0:
            print("No data available for the selected modality. Counting from var.")
            prep_df = var_df.copy()
            if groupby not in var_df.columns:
                raise ValueError(f"Column '{groupby}' not found in var data.")

            prep_df = var_df[[groupby, var_column]]
        else:
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

    def prep_id_bar(
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
        obs_df = self._get_obs(obs_column, groupby=groupby)
        var_df = self._get_var()
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

    def prep_intensity_hist(self, groupby: str, obs_column: str, bin_info: BinInfo) -> pd.DataFrame:
        """
        Calculates histogram bins for intensity distributions by group.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.
            bin_info: Precomputed bin metadata for binning.

        Returns:
            Histogram counts and frequencies per group and bin.
        """
        obs_df = self._get_obs(obs_column, groupby=groupby)
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

    def prep_intensity_bar(
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
        obs_df = self._get_obs(obs_column, groupby=groupby)
        orig_df = get_adata(self.mdata, self.modality).to_df().T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        join_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = join_df[[groupby, "_value"]]

        return prep_df

    def prep_intensity_simple_box(
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
        obs_df = self._get_obs(obs_column, groupby=groupby)
        orig_df = get_adata(self.mdata, self.modality).to_df().T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        join_df = melt_df.join(obs_df, on="_obs", how="left")

        prep_df = join_df[[groupby, "_value"]].groupby(groupby, observed=True).describe().droplevel(level=0, axis=1)
        prep_df.index = pd.CategoricalIndex(prep_df.index, categories=obs_df[groupby].unique())
        prep_df = prep_df.sort_index(axis=0)

        return prep_df

    def prep_missingness_step(
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
        obs = self._get_obs(obs_column)
        n_sample = obs.shape[0]

        # Prepare data
        orig_df = get_adata(self.mdata, self.modality).to_df()
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

    def prep_pca_scatter(
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
        obs = self._get_obs(obs_column, groupby=groupby)

        # Prepare data
        orig_df = pd.DataFrame(self.mdata[modality].obsm["X_pca"][pc_columns])
        join_df = orig_df.join(obs, how="left")
        join_df[groupby] = pd.Categorical(join_df[groupby], categories=obs[groupby].unique())

        return join_df

    def prep_umap_scatter(
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
        obs = self._get_obs(obs_column, groupby=groupby)

        # Prepare data
        orig_df = pd.DataFrame(self.mdata[modality].obsm["X_umap"][umap_columns])
        join_df = orig_df.join(obs, how="left")
        join_df[groupby] = pd.Categorical(join_df[groupby], categories=obs[groupby].unique())

        return join_df

    def prep_id_upset(
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

    def prep_intensity_correlation(self, groupby: str, obs_column: str) -> pd.DataFrame:
        """
        Computes pairwise Pearson correlations between grouped median profiles.

        Parameters:
            groupby: Observation column to group by.
            obs_column: Observation column to align with variables.

        Returns:
            Lower-triangular correlation matrix with NaNs above diagonal.
        """
        orig_df = self._get_data()
        obs_df = self._get_obs(obs_column, groupby=groupby)
        corrs_df = orig_df.groupby(obs_df[groupby], observed=True).median().T.corr(method="pearson")

        for x in range(corrs_df.shape[0]):
            for y in range(corrs_df.shape[1]):
                if x < y:
                    corrs_df.iloc[x, y] = np.nan

        corrs_df = corrs_df.sort_index(axis=0).sort_index(axis=1)

        return corrs_df
