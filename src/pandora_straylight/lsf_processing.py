from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from pandora_straylight.straylight_region import (
    StraylightRegionResult,
)


@dataclass
class NormalizedStraylightDistribution:
    """
    Normalized out-of-band response from one laser measurement.

    The normalization denominator is the integrated positive signal
    inside the selected in-band region.

    Two normalized distributions are stored:

    signed_distribution
        Retains negative dark-subtraction noise outside the in-band
        region.

    positive_distribution
        Replaces negative values with zero. This is the preferred
        distribution for later stray-light matrix construction.
    """

    wavelength_nm: float
    wavelength_occurrence: int
    routine_count: int

    source_peak_pixel: int

    ib_region_size: int
    ib_left_pixel: int
    ib_right_pixel: int

    normalization_value: float

    signed_distribution: np.ndarray
    positive_distribution: np.ndarray

    signed_sum: float
    positive_sum: float

    negative_pixel_count: int
    positive_pixel_count: int
    zero_pixel_count: int

    maximum_positive_value: float
    maximum_positive_pixel: int

    left_positive_sum: float
    right_positive_sum: float

    left_fraction: float
    right_fraction: float

    total_straylight_fraction: float


def validate_region_for_normalization(
    region: StraylightRegionResult,
) -> None:
    """
    Validate one stray-light region before normalization.
    """

    full_spectrum = np.asarray(
        region.full_spectrum,
        dtype=np.float64,
    )

    stray_light_spectrum = np.asarray(
        region.stray_light_spectrum,
        dtype=np.float64,
    )

    if full_spectrum.ndim != 1:
        raise ValueError(
            f"The full spectrum for {region.wavelength_nm:g} nm "
            "must be one-dimensional."
        )

    if full_spectrum.size != 2048:
        raise ValueError(
            f"Expected 2048 pixels for {region.wavelength_nm:g} nm, "
            f"but found {full_spectrum.size}."
        )

    if stray_light_spectrum.shape != full_spectrum.shape:
        raise ValueError(
            f"The stray-light spectrum shape does not match the "
            f"full spectrum for {region.wavelength_nm:g} nm."
        )

    if region.in_band_mask.shape != full_spectrum.shape:
        raise ValueError(
            f"The in-band mask has the wrong shape for "
            f"{region.wavelength_nm:g} nm."
        )

    if region.stray_light_mask.shape != full_spectrum.shape:
        raise ValueError(
            f"The stray-light mask has the wrong shape for "
            f"{region.wavelength_nm:g} nm."
        )

    if not np.all(np.isfinite(full_spectrum)):
        raise ValueError(
            f"The full spectrum for {region.wavelength_nm:g} nm "
            "contains non-finite values."
        )

    if not np.all(np.isfinite(stray_light_spectrum)):
        raise ValueError(
            f"The stray-light spectrum for "
            f"{region.wavelength_nm:g} nm contains non-finite values."
        )

    if not np.isfinite(region.in_band_sum):
        raise ValueError(
            f"The in-band normalization value for "
            f"{region.wavelength_nm:g} nm is not finite."
        )

    if region.in_band_sum <= 0:
        raise ValueError(
            f"The in-band normalization value for "
            f"{region.wavelength_nm:g} nm must be positive."
        )

    if not region.in_band_mask[region.peak_pixel_index]:
        raise ValueError(
            f"The detected peak for {region.wavelength_nm:g} nm "
            "is not inside the in-band region."
        )


def build_normalized_straylight_distribution(
    region: StraylightRegionResult,
) -> NormalizedStraylightDistribution:
    """
    Normalize one out-of-band laser response by integrated in-band
    signal.

    The in-band region is forced to zero in both output distributions.
    """

    validate_region_for_normalization(region)

    normalization_value = float(
        region.in_band_sum
    )

    signed_distribution = (
        np.asarray(
            region.stray_light_spectrum,
            dtype=np.float64,
        )
        / normalization_value
    )

    # Ensure that the direct laser region never contributes to the
    # stray-light response.
    signed_distribution = signed_distribution.copy()
    signed_distribution[region.in_band_mask] = 0.0

    positive_distribution = np.clip(
        signed_distribution,
        a_min=0.0,
        a_max=None,
    )

    positive_distribution[
        region.in_band_mask
    ] = 0.0

    signed_sum = float(
        np.sum(signed_distribution)
    )

    positive_sum = float(
        np.sum(positive_distribution)
    )

    negative_pixel_count = int(
        np.count_nonzero(
            signed_distribution < 0
        )
    )

    positive_pixel_count = int(
        np.count_nonzero(
            signed_distribution > 0
        )
    )

    zero_pixel_count = int(
        np.count_nonzero(
            signed_distribution == 0
        )
    )

    maximum_positive_pixel = int(
        np.argmax(positive_distribution)
    )

    maximum_positive_value = float(
        positive_distribution[
            maximum_positive_pixel
        ]
    )

    source_peak_pixel = int(
        region.peak_pixel_index
    )

    # Pixels to the left and right of the direct laser region.
    left_positive_sum = float(
        np.sum(
            positive_distribution[
                :region.ib_left_pixel
            ]
        )
    )

    right_positive_sum = float(
        np.sum(
            positive_distribution[
                region.ib_right_pixel + 1:
            ]
        )
    )

    if positive_sum > 0:
        left_fraction = (
            left_positive_sum
            / positive_sum
        )

        right_fraction = (
            right_positive_sum
            / positive_sum
        )
    else:
        left_fraction = float("nan")
        right_fraction = float("nan")

    return NormalizedStraylightDistribution(
        wavelength_nm=region.wavelength_nm,
        wavelength_occurrence=(
            region.wavelength_occurrence
        ),
        routine_count=region.routine_count,
        source_peak_pixel=source_peak_pixel,
        ib_region_size=region.ib_region_size,
        ib_left_pixel=region.ib_left_pixel,
        ib_right_pixel=region.ib_right_pixel,
        normalization_value=normalization_value,
        signed_distribution=signed_distribution,
        positive_distribution=positive_distribution,
        signed_sum=signed_sum,
        positive_sum=positive_sum,
        negative_pixel_count=negative_pixel_count,
        positive_pixel_count=positive_pixel_count,
        zero_pixel_count=zero_pixel_count,
        maximum_positive_value=maximum_positive_value,
        maximum_positive_pixel=maximum_positive_pixel,
        left_positive_sum=left_positive_sum,
        right_positive_sum=right_positive_sum,
        left_fraction=float(left_fraction),
        right_fraction=float(right_fraction),
        total_straylight_fraction=positive_sum,
    )


def build_all_normalized_straylight_distributions(
    regions: List[StraylightRegionResult],
) -> List[NormalizedStraylightDistribution]:
    """
    Build normalized stray-light distributions for all laser blocks.

    Duplicate wavelengths are intentionally preserved.
    """

    return [
        build_normalized_straylight_distribution(region)
        for region in regions
    ]