from mudata import MuData

from msmu import merge_mudata


# def test_merge_mudata_concatenates_modalities(sample_mudata):
#     merged = merge_mudata(
#         {
#             "set1": sample_mudata.copy(),
#             "set2": sample_mudata.copy(),
#         }
#     )

#     assert isinstance(merged, MuData)
#     assert merged["rna"].n_obs == sample_mudata["rna"].n_obs * 2
#     assert merged["protein"].n_vars == sample_mudata["protein"].n_vars
