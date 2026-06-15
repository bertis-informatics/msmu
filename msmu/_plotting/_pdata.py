"""
Module for preparing plotting data from MuData objects.
"""

from typing import cast

from mudata import MuData
import numpy as np
import pandas as pd

from .._utils._mudata import get_anndata_mod
from ..logging_utils import get_logger
from ._utils import BinInfo, get_bin_info, is_resolved_obs_groupby, obsm_embedding_to_frame, prepare_obs_frame

logger = get_logger(__name__)


class PlotData:
    def __init__(
        self,
        mdata: MuData,
        modality: str,
        layer: str | None = None,
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
        self.layer = layer
        self.kwargs = kwargs

    @staticmethod
    def _has_quantification(data: pd.DataFrame) -> bool:
        """Return whether the matrix contains any quantified values."""
        return bool(np.nansum(data) != 0)

    @staticmethod
    def _describe_by_group(prep_df: pd.DataFrame, groupby: str, categories) -> pd.DataFrame:
        """Summarize grouped numeric values into Plotly-ready box statistics."""
        described = prep_df.groupby(groupby, observed=True).describe().droplevel(level=0, axis=1)
        if isinstance(described, pd.Series):
            described = described.to_frame()
        described.index = pd.CategoricalIndex(described.index, categories=categories)
        return described.sort_index(axis=0)

    @staticmethod
    def _build_binned_frame(
        grouped: pd.DataFrame,
        *,
        group_categories,
        bin_info: BinInfo,
        denominator: int,
    ) -> pd.DataFrame:
        """Convert grouped bin counts into the long-form histogram payload expected by plot builders."""

        n_bins = len(bin_info["labels"])
        grouped = grouped.loc[grouped.sum(axis=1) > 0, :].copy()
        grouped.index = pd.CategoricalIndex(grouped.index, categories=group_categories)
        grouped = pd.DataFrame(grouped.sort_index(axis=0))

        bin_counts = np.asarray(grouped.to_numpy(dtype=float), dtype=float).reshape(-1)
        bin_freqs = bin_counts / denominator
        bin_names = grouped.index.get_level_values(0).repeat(n_bins).tolist()

        prepped = pd.DataFrame(
            {
                "center": bin_info["centers"] * len(grouped),
                "label": bin_info["labels"] * len(grouped),
                "count": bin_counts,
                "frequency": bin_freqs,
                "name": bin_names,
            }
        )
        prepped["name"] = pd.Categorical(prepped["name"], categories=group_categories)

        return prepped

    def _resolve_var_grouping_inputs(
        self,
        groupby: str,
        obs_column: str,
    ) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Resolve common inputs for variable-oriented plot preparation."""
        groupby_type, groupby_column = self._get_groupby_column(groupby, obs_column)
        obs_df = self._get_obs(obs_column, groupby=groupby_column if groupby_type == "obs" else None)
        var_df = self._get_var(groupby=groupby_column if groupby_type == "var" else None)
        orig_df = self._get_data()

        return groupby_type, obs_df, var_df, orig_df

    def _get_present_var_records(
        self,
        *,
        groupby: str,
        obs_df: pd.DataFrame,
        orig_df: pd.DataFrame,
        value_df: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame:
        """Return variable records that are observed within each group."""
        merged_df = orig_df.notna().join(obs_df[groupby], how="left")
        merged_df = merged_df.groupby(groupby, observed=True).any()

        melt_df = merged_df.stack().reset_index()
        melt_df.columns = [groupby, "_var", "_exists"]

        if isinstance(value_df, pd.Series):
            value_df = value_df.to_frame()

        prep_df = melt_df.merge(value_df, left_on="_var", right_index=True)
        prep_df = prep_df.loc[prep_df["_exists"] > 0, :]
        kept_columns = [column for column in prep_df.columns if column not in {"_var", "_exists"}]

        return prep_df.loc[:, kept_columns]

    def _get_intensity_long(
        self,
        *,
        groupby: str,
        obs_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return long-form intensity values joined with observation metadata."""
        obs_df = self._get_obs(obs_column, groupby=groupby)
        orig_df = self._get_data().T

        melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
        join_df = melt_df.join(obs_df, on="_obs", how="left")

        return obs_df, join_df

    def _get_groupby_column(self, groupby: str, obs_column: str) -> tuple[str, str]:
        """
        Resolves the grouping column from kwargs or uses the provided default.

        Parameters:
            groupby: Default grouping column name.
            obs_column: Observation column used to resolve plotting fallbacks.

        Returns:
            Tuple indicating whether the column is from 'obs' or 'var', and the column name.
        """

        if groupby in self.mdata.obs.columns:
            key = "obs"
        elif groupby == obs_column and is_resolved_obs_groupby(self.mdata, groupby, obs_column):
            key = "obs"
        elif groupby in get_anndata_mod(self.mdata, self.modality).var.columns:
            key = "var"
        elif is_resolved_obs_groupby(self.mdata, groupby, obs_column):
            key = "obs"
        else:
            raise ValueError(f"Column '{groupby}' not found in obs or var data.")

        return key, groupby

    def _get_data(self) -> pd.DataFrame:
        """
        Retrieves the expression/intensity DataFrame for the current modality.

        Returns:
            Copy of the modality's data matrix as a DataFrame.
        """
        adata = get_anndata_mod(self.mdata, self.modality).copy()

        if self.layer is not None:
            if self.layer not in adata.layers:
                raise ValueError(f"Layer '{self.layer}' not found in modality '{self.modality}'.")
            data = pd.DataFrame(adata.layers[self.layer], index=adata.obs_names, columns=adata.var_names)
        else:
            data = adata.to_df()

        return data

    def _get_var(self, groupby: str | None = None) -> pd.DataFrame:
        """
        Retrieves the variable metadata for the current modality.

        Returns:
            Copy of the modality's `var` table.
        """
        var_df = cast(pd.DataFrame, get_anndata_mod(self.mdata, self.modality).var.copy())

        if groupby and groupby in var_df.columns:
            if not isinstance(var_df[groupby].dtype, pd.CategoricalDtype):
                var_df[groupby] = pd.Categorical(var_df[groupby], categories=var_df[groupby].unique())
            else:
                var_df[groupby] = var_df[groupby].cat.remove_unused_categories()

        return var_df

    def _get_varm(self, column: str) -> pd.DataFrame:
        """
        Retrieves a varm column and merges it with `var` for plotting.

        Parameters:
            column: Name of the varm column to merge with `var`.

        Returns:
            Concatenated `var` and selected varm DataFrame.
        """
        var_df: pd.DataFrame = self._get_var()
        varm_df: pd.DataFrame = pd.DataFrame(get_anndata_mod(self.mdata, self.modality).varm[column].copy())

        return pd.concat([var_df, varm_df], axis=1)

    def _get_obs(self, obs_column: str, groupby: str | None = None) -> pd.DataFrame:
        """
        Retrieves observation metadata sorted and cast to categorical.

        Parameters:
            obs_column: Observation column used for ordering and grouping.

        Returns:
            Observation DataFrame with categorical ordering applied.
        """
        return prepare_obs_frame(self.mdata, obs_column, groupby=groupby)

    def _get_bin_info(
        self,
        data: pd.DataFrame | pd.Series | np.ndarray,
        bins: int,
    ) -> BinInfo:
        """
        Computes histogram bin metadata for numeric intensity data.

        Parameters:
            data: Numeric data for binning.
            bins: Number of bins to divide the data into.

        Returns:
            Bin width, edges, centers, and labels.
        """
        values = np.asarray(data, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Cannot compute bin info for empty data.")

        return get_bin_info(values, bins)

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
                logger.debug("No data available for the selected modality. Counting from var.")
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
        groupby_type, obs_df, var_df, orig_df = self._resolve_var_grouping_inputs(groupby, obs_column)

        if not self._has_quantification(orig_df) or groupby_type == "var":
            categories = var_df[groupby].unique()
            prep_df = var_df[[groupby, var_column]].groupby(groupby, observed=True).value_counts().reset_index()
            prep_df[groupby] = pd.Categorical(prep_df[groupby], categories=categories)
            prep_df = prep_df.sort_values(groupby).reset_index(drop=True)
        else:
            prep_df = self._get_present_var_records(
                groupby=groupby,
                obs_df=obs_df,
                orig_df=orig_df,
                value_df=var_df[[var_column]],
            )
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
        groupby_type, obs_df, var_df, orig_df = self._resolve_var_grouping_inputs(groupby, obs_column)

        if not self._has_quantification(orig_df) or groupby_type == "var":
            prep_df: pd.DataFrame = var_df.loc[:, [groupby, var_column]]
        else:
            prep_df = self._get_present_var_records(
                groupby=groupby,
                obs_df=obs_df,
                orig_df=orig_df,
                value_df=var_df[[var_column]],
            )

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
        groupby_type, obs_df, var_df, orig_df = self._resolve_var_grouping_inputs(groupby, obs_column)
        groupby_categories = obs_df[groupby].unique() if groupby_type == "obs" else var_df[groupby].unique()

        if not self._has_quantification(orig_df) or groupby_type == "var":
            prep_df: pd.DataFrame = var_df.loc[:, [groupby, var_column]]
        else:
            prep_df = self._get_present_var_records(
                groupby=groupby,
                obs_df=obs_df,
                orig_df=orig_df,
                value_df=var_df[var_column],
            )

        return self._describe_by_group(prep_df, groupby, groupby_categories)

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
        groupby_type, obs_df, var_df, orig_df = self._resolve_var_grouping_inputs(groupby, obs_column)
        group_categories = obs_df[groupby].unique() if groupby_type == "obs" else var_df[groupby].unique()

        if not self._has_quantification(orig_df) or groupby_type == "var":
            prep_df: pd.DataFrame = var_df.loc[:, [groupby, var_column]]
        else:
            prep_df = self._get_present_var_records(
                groupby=groupby,
                obs_df=obs_df,
                orig_df=orig_df,
                value_df=var_df[var_column],
            )

        prep_df["_bin_"] = pd.cut(
            prep_df[var_column],
            bins=bin_info["edges"],
            labels=bin_info["labels"],
            include_lowest=True,
        )

        grouped = prep_df.groupby([groupby, "_bin_"], observed=False).size().unstack(fill_value=0)
        if isinstance(grouped, pd.Series):
            grouped = grouped.to_frame()
        elif not isinstance(grouped, pd.DataFrame):
            grouped = pd.DataFrame(grouped)
        return self._build_binned_frame(
            grouped,
            group_categories=group_categories,
            bin_info=bin_info,
            denominator=prep_df.shape[0],
        )

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
        groupby_type, groupby_column = self._get_groupby_column(groupby, obs_column)
        obs_df = self._get_obs(obs_column, groupby=groupby_column if groupby_type == "obs" else None)
        var_df = self._get_var(groupby=groupby_column if groupby_type == "var" else None)
        orig_df = self._get_data()

        if np.nansum(orig_df) == 0 or groupby_type == "var":
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
        obs_df, join_df = self._get_intensity_long(groupby=groupby, obs_column=obs_column)

        if join_df.empty:
            raise ValueError("No data available for the selected modality.")

        join_df["_bin_"] = pd.cut(
            join_df["_value"],
            bins=bin_info["edges"],
            labels=bin_info["labels"],
            include_lowest=True,
        )

        grouped = join_df.groupby([groupby, "_bin_"], observed=False).size().unstack(fill_value=0)
        if isinstance(grouped, pd.Series):
            grouped = grouped.to_frame()
        elif not isinstance(grouped, pd.DataFrame):
            grouped = pd.DataFrame(grouped)
        return self._build_binned_frame(
            grouped,
            group_categories=obs_df[groupby].unique(),
            bin_info=bin_info,
            denominator=join_df.shape[0],
        )

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
        _, join_df = self._get_intensity_long(groupby=groupby, obs_column=obs_column)
        return pd.DataFrame(join_df.loc[:, [groupby, "_value"]])

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
        obs_df, join_df = self._get_intensity_long(groupby=groupby, obs_column=obs_column)
        prep_df = pd.DataFrame(join_df.loc[:, [groupby, "_value"]])
        return self._describe_by_group(prep_df, groupby, obs_df[groupby].unique())

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
        orig_df = self._get_data()
        sum_list = orig_df.isna().sum(axis=0)

        count_list = sum_list.value_counts().sort_index().cumsum()
        boundary_counts = pd.Series(
            [0, int(orig_df.shape[1])],
            index=pd.Index([0, int(n_sample)]),
            name=count_list.name,
        )
        count_list = pd.concat([count_list, boundary_counts]).groupby(level=0).max().sort_index()
        count_list.name = "count"

        prep_df = pd.DataFrame(count_list).reset_index(names="missingness")
        prep_df["ratio"] = prep_df["count"] / np.max(prep_df["count"]) * 100
        prep_df["missingness"] = prep_df["missingness"] / n_sample * 100
        prep_df["name"] = "Missingness"

        return prep_df

    def prep_embedding_scatter(
        self,
        modality: str,
        groupby: str,
        columns: list[str],
        obs_column: str,
        key: str,
    ) -> pd.DataFrame:
        """Join embedding coordinates with resolved observation metadata."""
        obs = self._get_obs(obs_column, groupby=groupby)

        orig_df = obsm_embedding_to_frame(self.mdata, modality, key, columns)
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
        binary_labels = ["".join(orig_df[column].astype(str).tolist()) for column in orig_df.columns]
        df_binary = pd.Series(binary_labels, index=orig_df.columns)

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
