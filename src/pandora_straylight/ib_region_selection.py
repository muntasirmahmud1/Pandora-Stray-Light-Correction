from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from pandora_straylight.laser_extraction import (
    ExtractedLaserSpectrum,
)
from pandora_straylight.straylight_region import (
    build_straylight_region,
)
from pandora_straylight.lsf_processing import (
    build_normalized_straylight_distribution,
)


@dataclass
class IBRegionSensitivityResult:
    """
    Sensitivity metrics for one laser measurement and one in-band size.
    """

    wavelength_nm: float
    wavelength_occurrence: int
    routine_count: int

    source_peak_pixel: int

    ib_region_size: int
    ib_left_pixel: int
    ib_right_pixel: int

    total_positive_fraction: float
    total_signed_fraction: float

    left_positive_fraction: float
    right_positive_fraction: float

    adjacent_left_sum: float
    adjacent_right_sum: float
    adjacent_total_sum: float

    adjacent_fraction_of_straylight: float

    maximum_out_of_band_value: float
    maximum_out_of_band_pixel: int
    maximum_distance_from_ib_region: int

    far_field_sum: float
    far_field_fraction_of_straylight: float


def distance_from_ib_region(
    pixel: int,
    ib_left_pixel: int,
    ib_right_pixel: int,
) -> int:
    """
    Calculate the distance of a pixel from the nearest in-band boundary.

    Returns
    -------
    0
        Pixel is inside the in-band region.

    1
        Pixel is immediately adjacent to the in-band region.
    """

    if pixel < ib_left_pixel:
        return ib_left_pixel - pixel

    if pixel > ib_right_pixel:
        return pixel - ib_right_pixel

    return 0


def evaluate_one_ib_region_size(
    extracted_spectrum: ExtractedLaserSpectrum,
    ib_region_size: int,
    adjacent_width: int = 5,
    far_field_distance: int = 30,
) -> IBRegionSensitivityResult:
    """
    Evaluate one in-band region size for one laser spectrum.

    Parameters
    ----------
    extracted_spectrum
        Dark-corrected laser spectrum.

    ib_region_size
        Total number of pixels in the in-band region.

    adjacent_width
        Number of out-of-band pixels examined immediately outside
        each in-band boundary.

    far_field_distance
        Out-of-band pixels farther than this distance from the
        in-band region are classified as far-field stray light.
    """

    if adjacent_width <= 0:
        raise ValueError(
            "adjacent_width must be greater than zero."
        )

    if far_field_distance < 0:
        raise ValueError(
            "far_field_distance cannot be negative."
        )

    region = build_straylight_region(
        extracted_spectrum=extracted_spectrum,
        ib_region_size=ib_region_size,
    )

    distribution = (
        build_normalized_straylight_distribution(
            region
        )
    )

    positive = distribution.positive_distribution

    left_start = max(
        0,
        region.ib_left_pixel - adjacent_width,
    )

    left_stop = region.ib_left_pixel

    right_start = region.ib_right_pixel + 1

    right_stop = min(
        positive.size,
        right_start + adjacent_width,
    )

    adjacent_left_sum = float(
        np.sum(
            positive[
                left_start:left_stop
            ]
        )
    )

    adjacent_right_sum = float(
        np.sum(
            positive[
                right_start:right_stop
            ]
        )
    )

    adjacent_total_sum = (
        adjacent_left_sum
        + adjacent_right_sum
    )

    total_positive_fraction = float(
        distribution.positive_sum
    )

    if total_positive_fraction > 0:
        adjacent_fraction_of_straylight = (
            adjacent_total_sum
            / total_positive_fraction
        )
    else:
        adjacent_fraction_of_straylight = float("nan")

    maximum_out_of_band_pixel = int(
        np.argmax(positive)
    )

    maximum_out_of_band_value = float(
        positive[
            maximum_out_of_band_pixel
        ]
    )

    maximum_distance = distance_from_ib_region(
        pixel=maximum_out_of_band_pixel,
        ib_left_pixel=region.ib_left_pixel,
        ib_right_pixel=region.ib_right_pixel,
    )

    detector_pixels = np.arange(
        positive.size
    )

    left_distance = (
        region.ib_left_pixel
        - detector_pixels
    )

    right_distance = (
        detector_pixels
        - region.ib_right_pixel
    )

    distance_from_region = np.maximum(
        left_distance,
        right_distance,
    )

    far_field_mask = (
        region.stray_light_mask
        & (
            distance_from_region
            > far_field_distance
        )
    )

    far_field_sum = float(
        np.sum(
            positive[
                far_field_mask
            ]
        )
    )

    if total_positive_fraction > 0:
        far_field_fraction_of_straylight = (
            far_field_sum
            / total_positive_fraction
        )
    else:
        far_field_fraction_of_straylight = float("nan")

    return IBRegionSensitivityResult(
        wavelength_nm=(
            extracted_spectrum.wavelength_nm
        ),
        wavelength_occurrence=(
            extracted_spectrum.wavelength_occurrence
        ),
        routine_count=(
            extracted_spectrum.routine_count
        ),
        source_peak_pixel=(
            extracted_spectrum.peak_pixel_index
        ),
        ib_region_size=ib_region_size,
        ib_left_pixel=region.ib_left_pixel,
        ib_right_pixel=region.ib_right_pixel,
        total_positive_fraction=(
            total_positive_fraction
        ),
        total_signed_fraction=float(
            distribution.signed_sum
        ),
        left_positive_fraction=float(
            distribution.left_fraction
        ),
        right_positive_fraction=float(
            distribution.right_fraction
        ),
        adjacent_left_sum=adjacent_left_sum,
        adjacent_right_sum=adjacent_right_sum,
        adjacent_total_sum=adjacent_total_sum,
        adjacent_fraction_of_straylight=float(
            adjacent_fraction_of_straylight
        ),
        maximum_out_of_band_value=(
            maximum_out_of_band_value
        ),
        maximum_out_of_band_pixel=(
            maximum_out_of_band_pixel
        ),
        maximum_distance_from_ib_region=(
            maximum_distance
        ),
        far_field_sum=far_field_sum,
        far_field_fraction_of_straylight=float(
            far_field_fraction_of_straylight
        ),
    )


def evaluate_ib_region_sizes(
    extracted_spectra: List[ExtractedLaserSpectrum],
    ib_region_sizes: Iterable[int],
    adjacent_width: int = 5,
    far_field_distance: int = 30,
) -> List[IBRegionSensitivityResult]:
    """
    Evaluate multiple in-band region sizes for all laser blocks.

    Duplicate laser wavelengths remain separate.
    """

    sizes = list(ib_region_sizes)

    if not sizes:
        raise ValueError(
            "At least one in-band region size is required."
        )

    results: List[IBRegionSensitivityResult] = []

    for extracted_spectrum in extracted_spectra:
        for ib_region_size in sizes:
            result = evaluate_one_ib_region_size(
                extracted_spectrum=extracted_spectrum,
                ib_region_size=ib_region_size,
                adjacent_width=adjacent_width,
                far_field_distance=far_field_distance,
            )

            results.append(result)

    return results


def calculate_median_metrics_by_size(
    results: List[IBRegionSensitivityResult],
) -> dict[int, dict[str, float]]:
    """
    Calculate median sensitivity metrics across all laser blocks.
    """

    sizes = sorted(
        {
            result.ib_region_size
            for result in results
        }
    )

    summary: dict[int, dict[str, float]] = {}

    for size in sizes:
        size_results = [
            result
            for result in results
            if result.ib_region_size == size
        ]

        total_fractions = np.asarray(
            [
                result.total_positive_fraction
                for result in size_results
            ],
            dtype=np.float64,
        )

        adjacent_fractions = np.asarray(
            [
                result.adjacent_fraction_of_straylight
                for result in size_results
            ],
            dtype=np.float64,
        )

        far_field_fractions = np.asarray(
            [
                result.far_field_fraction_of_straylight
                for result in size_results
            ],
            dtype=np.float64,
        )

        maximum_values = np.asarray(
            [
                result.maximum_out_of_band_value
                for result in size_results
            ],
            dtype=np.float64,
        )

        summary[size] = {
            "median_total_positive_fraction": float(
                np.nanmedian(total_fractions)
            ),
            "minimum_total_positive_fraction": float(
                np.nanmin(total_fractions)
            ),
            "maximum_total_positive_fraction": float(
                np.nanmax(total_fractions)
            ),
            "median_adjacent_fraction": float(
                np.nanmedian(adjacent_fractions)
            ),
            "median_far_field_fraction": float(
                np.nanmedian(far_field_fractions)
            ),
            "median_maximum_value": float(
                np.nanmedian(maximum_values)
            ),
        }

    return summary