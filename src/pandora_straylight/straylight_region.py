from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from pandora_straylight.laser_extraction import (
    ExtractedLaserSpectrum,
)


@dataclass
class StraylightRegionResult:
    """
    In-band and out-of-band regions for one laser spectrum.

    The in-band region represents the direct laser signal.

    Everything outside the in-band region is treated as the measured
    stray-light distribution for this laser.
    """

    wavelength_nm: float
    wavelength_occurrence: int
    routine_count: int

    peak_pixel_index: int

    ib_region_size: int
    ib_left_pixel: int
    ib_right_pixel: int

    in_band_mask: np.ndarray
    stray_light_mask: np.ndarray

    full_spectrum: np.ndarray
    in_band_spectrum: np.ndarray
    stray_light_spectrum: np.ndarray

    in_band_sum: float
    stray_light_sum: float
    total_positive_sum: float

    stray_light_fraction: float

    in_band_pixel_count: int
    stray_light_pixel_count: int


def validate_ib_region_size(
    ib_region_size: int,
) -> None:
    """
    Validate the in-band region size.

    In this implementation, ib_region_size is the total number of pixels
    retained around the detected peak.

    Examples
    --------
    ib_region_size = 21

    For a peak at pixel 870, the in-band region is approximately:

        860 through 880

    including 21 detector pixels.
    """

    if not isinstance(ib_region_size, int):
        raise TypeError(
            "ib_region_size must be an integer."
        )

    if ib_region_size <= 0:
        raise ValueError(
            "ib_region_size must be greater than zero."
        )

    if ib_region_size > 2048:
        raise ValueError(
            "ib_region_size cannot exceed 2048 pixels."
        )


def calculate_ib_bounds(
    peak_pixel_index: int,
    ib_region_size: int,
    spectrum_size: int = 2048,
) -> tuple[int, int]:
    """
    Calculate inclusive in-band boundaries around the peak.

    The returned right boundary is inclusive.

    For odd sizes, the peak is exactly centered.

    For even sizes, one additional pixel is placed on the right side.
    """

    validate_ib_region_size(ib_region_size)

    if peak_pixel_index < 0 or peak_pixel_index >= spectrum_size:
        raise ValueError(
            f"Peak pixel {peak_pixel_index} is outside the valid "
            f"range 0 to {spectrum_size - 1}."
        )

    left_half = (ib_region_size - 1) // 2
    right_half = ib_region_size - left_half - 1

    left_pixel = peak_pixel_index - left_half
    right_pixel = peak_pixel_index + right_half

    if left_pixel < 0:
        shift = -left_pixel
        left_pixel += shift
        right_pixel += shift

    if right_pixel >= spectrum_size:
        shift = right_pixel - spectrum_size + 1
        left_pixel -= shift
        right_pixel -= shift

    if left_pixel < 0 or right_pixel >= spectrum_size:
        raise ValueError(
            "Could not construct the requested in-band region "
            "inside the detector limits."
        )

    actual_size = right_pixel - left_pixel + 1

    if actual_size != ib_region_size:
        raise RuntimeError(
            f"Requested an in-band size of {ib_region_size}, "
            f"but constructed {actual_size} pixels."
        )

    return left_pixel, right_pixel


def keep_positive_values(
    spectrum: np.ndarray,
) -> np.ndarray:
    """
    Return a copy where negative values are replaced by zero.

    The original corrected spectrum remains unchanged.

    Positive-only values are used only when calculating integrated
    signal fractions. This prevents negative dark-noise fluctuations
    from cancelling positive optical signal.
    """

    spectrum = np.asarray(
        spectrum,
        dtype=np.float64,
    )

    return np.clip(
        spectrum,
        a_min=0.0,
        a_max=None,
    )


def build_straylight_region(
    extracted_spectrum: ExtractedLaserSpectrum,
    ib_region_size: int,
) -> StraylightRegionResult:
    """
    Separate one corrected laser spectrum into:

    1. in-band laser signal
    2. out-of-band stray light
    """

    validate_ib_region_size(ib_region_size)

    full_spectrum = np.asarray(
        extracted_spectrum.corrected_counts_per_cycle,
        dtype=np.float64,
    )

    if full_spectrum.size != 2048:
        raise ValueError(
            f"Expected 2048 pixels for "
            f"{extracted_spectrum.wavelength_nm:g} nm, "
            f"but found {full_spectrum.size}."
        )

    if not np.all(np.isfinite(full_spectrum)):
        raise ValueError(
            f"The {extracted_spectrum.wavelength_nm:g} nm spectrum "
            "contains non-finite values."
        )

    peak_pixel_index = int(
        extracted_spectrum.peak_pixel_index
    )

    ib_left_pixel, ib_right_pixel = calculate_ib_bounds(
        peak_pixel_index=peak_pixel_index,
        ib_region_size=ib_region_size,
        spectrum_size=full_spectrum.size,
    )

    in_band_mask = np.zeros(
        full_spectrum.size,
        dtype=bool,
    )

    in_band_mask[
        ib_left_pixel:ib_right_pixel + 1
    ] = True

    stray_light_mask = ~in_band_mask

    in_band_spectrum = np.zeros_like(
        full_spectrum
    )

    stray_light_spectrum = np.zeros_like(
        full_spectrum
    )

    in_band_spectrum[in_band_mask] = (
        full_spectrum[in_band_mask]
    )

    stray_light_spectrum[stray_light_mask] = (
        full_spectrum[stray_light_mask]
    )

    positive_full = keep_positive_values(
        full_spectrum
    )

    positive_in_band = positive_full[
        in_band_mask
    ]

    positive_stray_light = positive_full[
        stray_light_mask
    ]

    in_band_sum = float(
        np.sum(positive_in_band)
    )

    stray_light_sum = float(
        np.sum(positive_stray_light)
    )

    total_positive_sum = float(
        np.sum(positive_full)
    )

    if total_positive_sum > 0:
        stray_light_fraction = (
            stray_light_sum
            / total_positive_sum
        )
    else:
        stray_light_fraction = float("nan")

    return StraylightRegionResult(
        wavelength_nm=extracted_spectrum.wavelength_nm,
        wavelength_occurrence=(
            extracted_spectrum.wavelength_occurrence
        ),
        routine_count=extracted_spectrum.routine_count,
        peak_pixel_index=peak_pixel_index,
        ib_region_size=ib_region_size,
        ib_left_pixel=ib_left_pixel,
        ib_right_pixel=ib_right_pixel,
        in_band_mask=in_band_mask,
        stray_light_mask=stray_light_mask,
        full_spectrum=full_spectrum.copy(),
        in_band_spectrum=in_band_spectrum,
        stray_light_spectrum=stray_light_spectrum,
        in_band_sum=in_band_sum,
        stray_light_sum=stray_light_sum,
        total_positive_sum=total_positive_sum,
        stray_light_fraction=float(
            stray_light_fraction
        ),
        in_band_pixel_count=int(
            np.count_nonzero(in_band_mask)
        ),
        stray_light_pixel_count=int(
            np.count_nonzero(stray_light_mask)
        ),
    )


def build_all_straylight_regions(
    extracted_spectra: List[ExtractedLaserSpectrum],
    ib_region_size: int,
) -> List[StraylightRegionResult]:
    """
    Build in-band and stray-light regions for all laser spectra.
    """

    return [
        build_straylight_region(
            extracted_spectrum=spectrum,
            ib_region_size=ib_region_size,
        )
        for spectrum in extracted_spectra
    ]