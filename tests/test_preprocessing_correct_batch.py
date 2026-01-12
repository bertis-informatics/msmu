import pytest
import numpy as np

from msmu._preprocessing._batch_correction import correct_batch_effect


def test_correct_batch_effect_gis_drops_samples(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="gis",
        rescale=False,
        category="batch",
        gis_samples=["gis1", "gis2"],
        drop_gis=True,
    )
    assert out["psm"].n_obs == 4  # 2 GIS samples dropped


def test_correct_batch_effect_gis_keep_samples(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="gis",
        rescale=False,
        category="batch",
        gis_samples=["gis1", "gis2"],
        drop_gis=False,
    )
    assert out["psm"].n_obs == 6  # 2 GIS samples kept


def test_correct_batch_effect_gis(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="gis",
        rescale=False,
        category="batch",
        gis_samples=["gis1", "gis2"],
        log_transformed=True,
        drop_gis=False,
    )
    assert np.allclose(
        out["psm"].X,
        np.array(
            [
                [-5.0, -4.0],
                [-3.0, -2.0],
                [0.0, 0.0],
                [-4.0, -4.0],
                [-2.0, -2.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_correct_batch_effect_gis_protein(simple_mdata_protein):
    out = correct_batch_effect(
        simple_mdata_protein,
        modality="protein",
        method="gis",
        rescale=False,
        category="batch",
        gis_samples=["gis1", "gis2"],
        log_transformed=True,
        drop_gis=False,
    )
    assert np.allclose(
        out["protein"].X,
        np.array(
            [
                [-4.0, -4.0],
                [-2.0, -2.0],
                [0.0, 0.0],
                [-4.0, -4.0],
                [-2.0, -2.0],
                [0.0, 0.0],
            ]
        ),
    )
    assert np.allclose(
        out["psm"].X,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        ),
    )


def test_correct_batch_effect_gis_rescale(simple_mdata):
    total_median = np.median(simple_mdata["psm"].X.flatten())
    answer = np.array(
        [
            [-5.0, -4.0],
            [-3.0, -2.0],
            [0.0, 0.0],
            [-4.0, -4.0],
            [-2.0, -2.0],
            [0.0, 0.0],
        ]
    )

    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="gis",
        rescale=True,
        category="batch",
        gis_samples=["gis1", "gis2"],
        log_transformed=True,
        drop_gis=False,
    )
    assert np.allclose(out["psm"].X, (answer + total_median))


def test_correct_batch_effect_median_center(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="median_center",
        rescale=False,
        category="batch",
        log_transformed=True,
    )
    assert np.allclose(
        out["psm"].X,
        np.array(
            [
                [-2.0, -2.0],
                [0.0, 0.0],
                [3.0, 2.0],
                [-2.0, -2.0],
                [0.0, 0.0],
                [2.0, 2.0],
            ]
        ),
    )


def test_correct_batch_effect_gis_missing_raises(simple_mdata):
    with pytest.raises(ValueError, match="as GIS not found in obs"):
        correct_batch_effect(
            simple_mdata,
            modality="psm",
            method="gis",
            rescale=False,
            category="batch",
            gis_samples=["gis4", "gis5"],
        )


def test_correct_batch_effect_invalid_method(simple_mdata):
    with pytest.raises(ValueError, match="not recognised"):
        correct_batch_effect(
            simple_mdata,
            modality="psm",
            category="batch",
            method="nope",
        )


def test_correct_batch_effect_combat(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="combat",
        rescale=False,
        category="batch",
        log_transformed=True,
    )
    assert out["psm"].X.shape == (6, 2)


def test_correct_batch_effect_continuous(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="continuous",
        rescale=False,
        category="batch",
        log_transformed=True,
    )
    assert out["psm"].X.shape == (6, 2)
