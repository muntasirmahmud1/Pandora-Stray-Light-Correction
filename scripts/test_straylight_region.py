from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# Project imports
# ============================================================

from pandora_straylight.l0_parser import load_pandora_l0
from pandora_straylight.laser_detection import detect_laser_blocks
from pandora_straylight.laser_extraction import (
    extract_all_laser_spectra,
)
from pandora_straylight.straylight_region import (
    build_all_straylight_regions,
)


# ============================================================
# Input and output paths
# ============================================================

CALIBRATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Pandora63s1_LabGSFC_20250311_L0.txt"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "straylight_region_diagnostics"
)


# ============================================================
# In-band region settings
# ============================================================

# Total number of pixels included in the in-band region.
#
# For example:
#
#     peak pixel = 870
#     IB_REGION_SIZE = 21
#
# gives an in-band region from pixel 860 through 880,
# including exactly 21 detector pixels.
IB_REGION_SIZE = 21


# ============================================================
# Plot settings
# ============================================================

# Normalize each laser spectrum by its corrected peak value.
#
# This makes the laser peak equal to 1.0 and makes spectra from
# different wavelengths directly comparable.
NORMALIZE_FOR_LOG_PLOTS = True


# Relative lower limit used for logarithmic plots.
#
# Because normalized spectra have a peak of 1.0, a value of 1e-7
# means that the displayed range extends seven orders of magnitude
# below the laser peak.
LOG_Y_MIN = 1e-7

LOG_Y_MAX = 2.0


# Number of pixels shown outside each in-band boundary in the
# peak-region diagnostic plots.
PEAK_ZOOM_EXTRA_PIXELS = 30


# ============================================================
# Plot preparation functions
# ============================================================

def normalize_by_peak(
    spectrum: np.ndarray,
    peak_value: float,
) -> np.ndarray:
    """
    Normalize a spectrum using its laser peak.

    The returned spectrum is a new array. The original detector
    values are not modified.
    """

    spectrum = np.asarray(
        spectrum,
        dtype=np.float64,
    )

    if not np.isfinite(peak_value):
        raise ValueError(
            "The peak value is not finite."
        )

    if peak_value <= 0:
        raise ValueError(
            "The peak value must be greater than zero for "
            "normalization."
        )

    return spectrum / peak_value


def prepare_log_plot_values(
    spectrum: np.ndarray,
    peak_value: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Prepare spectral values for logarithmic plotting.

    Logarithmic axes cannot display zero or negative values.
    Nonpositive values are therefore replaced with NaN so that
    Matplotlib does not draw them.

    This affects only the plotted copy. It does not alter any
    scientific data or calculations.
    """

    spectrum = np.asarray(
        spectrum,
        dtype=np.float64,
    ).copy()

    if normalize:
        spectrum = normalize_by_peak(
            spectrum=spectrum,
            peak_value=peak_value,
        )

    invalid_mask = (
        ~np.isfinite(spectrum)
        | (spectrum <= 0)
    )

    spectrum[invalid_mask] = np.nan

    return spectrum


def get_log_axis_label(
    normalized: bool,
) -> str:
    """
    Return the appropriate logarithmic y-axis label.
    """

    if normalized:
        return "Signal relative to laser peak"

    return "Corrected counts per cycle"


# ============================================================
# Full-spectrum plot
# ============================================================

def plot_full_spectrum(
    region,
    output_directory: Path,
) -> None:
    """
    Plot the complete corrected laser spectrum on a logarithmic axis.

    The main laser peak is normalized to 1.0 when
    NORMALIZE_FOR_LOG_PLOTS is True.
    """

    pixels = np.arange(
        region.full_spectrum.size
    )

    peak_value = float(
        region.full_spectrum[
            region.peak_pixel_index
        ]
    )

    plot_values = prepare_log_plot_values(
        spectrum=region.full_spectrum,
        peak_value=peak_value,
        normalize=NORMALIZE_FOR_LOG_PLOTS,
    )

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        pixels,
        plot_values,
        linewidth=1.0,
        label="Dark-corrected laser spectrum",
    )

    axis.axvline(
        region.peak_pixel_index,
        linestyle="--",
        linewidth=1.2,
        label="Detected peak",
    )

    axis.axvline(
        region.ib_left_pixel,
        linestyle=":",
        linewidth=1.5,
        label="In-band left boundary",
    )

    axis.axvline(
        region.ib_right_pixel,
        linestyle=":",
        linewidth=1.5,
        label="In-band right boundary",
    )

    axis.axvspan(
        region.ib_left_pixel,
        region.ib_right_pixel,
        alpha=0.2,
        label="In-band region",
    )

    axis.set_yscale("log")

    if NORMALIZE_FOR_LOG_PLOTS:
        axis.set_ylim(
            LOG_Y_MIN,
            LOG_Y_MAX,
        )

    axis.set_xlim(
        0,
        region.full_spectrum.size - 1,
    )

    axis.set_xlabel("Detector pixel")

    axis.set_ylabel(
        get_log_axis_label(
            NORMALIZE_FOR_LOG_PLOTS
        )
    )

    axis.set_title(
        f"{region.wavelength_nm:g} nm laser, "
        f"occurrence {region.wavelength_occurrence}\n"
        f"In-band size = {region.ib_region_size} pixels, "
        f"positive out-of-band fraction = "
        f"{100.0 * region.stray_light_fraction:.4f}%"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    filename = (
        f"laser_{region.wavelength_nm:g}nm"
        f"_occ{region.wavelength_occurrence}"
        f"_full_log.png"
    )

    figure.savefig(
        output_directory / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Linear peak-region plot
# ============================================================

def plot_peak_region_linear(
    region,
    output_directory: Path,
    extra_pixels: int = PEAK_ZOOM_EXTRA_PIXELS,
) -> None:
    """
    Plot a linear-scale zoom around the laser peak.

    Linear scale is retained here because it is useful for inspecting
    the peak shape and checking whether the selected in-band boundaries
    contain the direct laser signal.
    """

    plot_left = max(
        0,
        region.ib_left_pixel - extra_pixels,
    )

    plot_right = min(
        region.full_spectrum.size - 1,
        region.ib_right_pixel + extra_pixels,
    )

    pixels = np.arange(
        plot_left,
        plot_right + 1,
    )

    values = region.full_spectrum[
        plot_left:plot_right + 1
    ]

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.plot(
        pixels,
        values,
        marker=".",
        linewidth=1.0,
        label="Corrected spectrum",
    )

    axis.axvline(
        region.peak_pixel_index,
        linestyle="--",
        linewidth=1.2,
        label="Peak",
    )

    axis.axvline(
        region.ib_left_pixel,
        linestyle=":",
        linewidth=1.2,
        label="In-band left boundary",
    )

    axis.axvline(
        region.ib_right_pixel,
        linestyle=":",
        linewidth=1.2,
        label="In-band right boundary",
    )

    axis.axvspan(
        region.ib_left_pixel,
        region.ib_right_pixel,
        alpha=0.2,
        label="In-band region",
    )

    axis.set_xlabel("Detector pixel")
    axis.set_ylabel("Corrected counts per cycle")

    axis.set_title(
        f"{region.wavelength_nm:g} nm peak region — linear scale\n"
        f"Peak pixel = {region.peak_pixel_index}, "
        f"IB pixels = {region.ib_left_pixel}–"
        f"{region.ib_right_pixel}"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    filename = (
        f"laser_{region.wavelength_nm:g}nm"
        f"_occ{region.wavelength_occurrence}"
        f"_peak_zoom_linear.png"
    )

    figure.savefig(
        output_directory / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Logarithmic peak-region plot
# ============================================================

def plot_peak_region_log(
    region,
    output_directory: Path,
    extra_pixels: int = PEAK_ZOOM_EXTRA_PIXELS,
) -> None:
    """
    Plot a logarithmic zoom around the laser peak.

    This helps visualize the lower-intensity wings near the main peak.
    """

    plot_left = max(
        0,
        region.ib_left_pixel - extra_pixels,
    )

    plot_right = min(
        region.full_spectrum.size - 1,
        region.ib_right_pixel + extra_pixels,
    )

    pixels = np.arange(
        plot_left,
        plot_right + 1,
    )

    peak_value = float(
        region.full_spectrum[
            region.peak_pixel_index
        ]
    )

    values = prepare_log_plot_values(
        spectrum=region.full_spectrum[
            plot_left:plot_right + 1
        ],
        peak_value=peak_value,
        normalize=NORMALIZE_FOR_LOG_PLOTS,
    )

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.plot(
        pixels,
        values,
        marker=".",
        linewidth=1.0,
        label="Corrected spectrum",
    )

    axis.axvline(
        region.peak_pixel_index,
        linestyle="--",
        linewidth=1.2,
        label="Peak",
    )

    axis.axvline(
        region.ib_left_pixel,
        linestyle=":",
        linewidth=1.2,
        label="In-band left boundary",
    )

    axis.axvline(
        region.ib_right_pixel,
        linestyle=":",
        linewidth=1.2,
        label="In-band right boundary",
    )

    axis.axvspan(
        region.ib_left_pixel,
        region.ib_right_pixel,
        alpha=0.2,
        label="In-band region",
    )

    axis.set_yscale("log")

    if NORMALIZE_FOR_LOG_PLOTS:
        axis.set_ylim(
            LOG_Y_MIN,
            LOG_Y_MAX,
        )

    axis.set_xlabel("Detector pixel")

    axis.set_ylabel(
        get_log_axis_label(
            NORMALIZE_FOR_LOG_PLOTS
        )
    )

    axis.set_title(
        f"{region.wavelength_nm:g} nm peak region — log scale\n"
        f"Peak pixel = {region.peak_pixel_index}, "
        f"IB pixels = {region.ib_left_pixel}–"
        f"{region.ib_right_pixel}"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    filename = (
        f"laser_{region.wavelength_nm:g}nm"
        f"_occ{region.wavelength_occurrence}"
        f"_peak_zoom_log.png"
    )

    figure.savefig(
        output_directory / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Out-of-band stray-light plot
# ============================================================

def plot_stray_light_only(
    region,
    output_directory: Path,
) -> None:
    """
    Plot only the out-of-band signal using a logarithmic y-axis.

    The in-band pixels are represented by NaN, producing a visible
    gap around the main laser peak.

    Nonpositive dark-subtracted values are also hidden because they
    cannot be represented on a logarithmic axis.
    """

    pixels = np.arange(
        region.full_spectrum.size
    )

    peak_value = float(
        region.full_spectrum[
            region.peak_pixel_index
        ]
    )

    stray_light_for_plot = prepare_log_plot_values(
        spectrum=region.stray_light_spectrum,
        peak_value=peak_value,
        normalize=NORMALIZE_FOR_LOG_PLOTS,
    )

    stray_light_for_plot[
        region.in_band_mask
    ] = np.nan

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        pixels,
        stray_light_for_plot,
        linewidth=1.0,
        label="Out-of-band signal",
    )

    axis.axvline(
        region.peak_pixel_index,
        linestyle="--",
        linewidth=1.0,
        label="Laser peak position",
    )

    axis.axvline(
        region.ib_left_pixel,
        linestyle=":",
        linewidth=1.2,
        label="In-band left boundary",
    )

    axis.axvline(
        region.ib_right_pixel,
        linestyle=":",
        linewidth=1.2,
        label="In-band right boundary",
    )

    axis.axvspan(
        region.ib_left_pixel,
        region.ib_right_pixel,
        alpha=0.2,
        label="Removed in-band region",
    )

    axis.set_yscale("log")

    if NORMALIZE_FOR_LOG_PLOTS:
        axis.set_ylim(
            LOG_Y_MIN,
            LOG_Y_MAX,
        )

    axis.set_xlim(
        0,
        region.full_spectrum.size - 1,
    )

    axis.set_xlabel("Detector pixel")

    axis.set_ylabel(
        get_log_axis_label(
            NORMALIZE_FOR_LOG_PLOTS
        )
    )

    axis.set_title(
        f"{region.wavelength_nm:g} nm out-of-band signal\n"
        f"Occurrence {region.wavelength_occurrence}, "
        f"positive out-of-band fraction = "
        f"{100.0 * region.stray_light_fraction:.4f}%"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    filename = (
        f"laser_{region.wavelength_nm:g}nm"
        f"_occ{region.wavelength_occurrence}"
        f"_straylight_log.png"
    )

    figure.savefig(
        output_directory / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Combined normalized out-of-band comparison
# ============================================================

def plot_all_stray_light_regions(
    regions,
    output_directory: Path,
) -> None:
    """
    Plot normalized out-of-band distributions from all laser blocks.

    Duplicate wavelengths, including both 488 nm measurements, are
    intentionally shown separately.
    """

    pixels = np.arange(2048)

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    for region in regions:
        peak_value = float(
            region.full_spectrum[
                region.peak_pixel_index
            ]
        )

        plot_values = prepare_log_plot_values(
            spectrum=region.stray_light_spectrum,
            peak_value=peak_value,
            normalize=True,
        )

        plot_values[
            region.in_band_mask
        ] = np.nan

        label = (
            f"{region.wavelength_nm:g} nm "
            f"(occ. {region.wavelength_occurrence})"
        )

        axis.plot(
            pixels,
            plot_values,
            linewidth=0.9,
            label=label,
        )

    axis.set_yscale("log")

    axis.set_ylim(
        LOG_Y_MIN,
        LOG_Y_MAX,
    )

    axis.set_xlim(
        0,
        2047,
    )

    axis.set_xlabel("Detector pixel")
    axis.set_ylabel("Signal relative to laser peak")

    axis.set_title(
        "Normalized out-of-band distributions for all laser blocks"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(
        ncol=2,
        fontsize=8,
    )

    figure.tight_layout()

    filename = (
        "all_lasers_normalized_straylight_log.png"
    )

    figure.savefig(
        output_directory / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Main test
# ============================================================

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    l0_file = load_pandora_l0(
        CALIBRATION_FILE
    )

    laser_blocks = detect_laser_blocks(
        l0_file
    )

    extracted_spectra = extract_all_laser_spectra(
        laser_blocks
    )

    regions = build_all_straylight_regions(
        extracted_spectra=extracted_spectra,
        ib_region_size=IB_REGION_SIZE,
    )

    print("=" * 125)
    print("In-band and stray-light region diagnostics")
    print("=" * 125)

    print(
        f"IB region size: "
        f"{IB_REGION_SIZE} pixels"
    )

    print(
        f"Normalize log plots: "
        f"{NORMALIZE_FOR_LOG_PLOTS}"
    )

    print(
        f"Normalized log y-range: "
        f"{LOG_Y_MIN:g} to {LOG_Y_MAX:g}"
    )

    print()

    print(
        f"{'Wave':<9}"
        f"{'Occ':<6}"
        f"{'Peak':<8}"
        f"{'IB left':<10}"
        f"{'IB right':<11}"
        f"{'IB pixels':<11}"
        f"{'SL pixels':<11}"
        f"{'IB sum':<16}"
        f"{'SL sum':<16}"
        f"{'SL fraction':<14}"
    )

    print("-" * 125)

    for region in regions:
        print(
            f"{region.wavelength_nm:<9g}"
            f"{region.wavelength_occurrence:<6}"
            f"{region.peak_pixel_index:<8}"
            f"{region.ib_left_pixel:<10}"
            f"{region.ib_right_pixel:<11}"
            f"{region.in_band_pixel_count:<11}"
            f"{region.stray_light_pixel_count:<11}"
            f"{region.in_band_sum:<16.3f}"
            f"{region.stray_light_sum:<16.3f}"
            f"{100.0 * region.stray_light_fraction:<13.6f}%"
        )

        # ----------------------------------------------------
        # Validation tests
        # ----------------------------------------------------

        assert (
            region.in_band_pixel_count
            == IB_REGION_SIZE
        )

        assert (
            region.in_band_pixel_count
            + region.stray_light_pixel_count
            == 2048
        )

        assert np.all(
            region.in_band_mask
            != region.stray_light_mask
        )

        assert not np.any(
            region.in_band_mask
            & region.stray_light_mask
        )

        assert np.allclose(
            region.in_band_spectrum
            + region.stray_light_spectrum,
            region.full_spectrum,
        )

        assert (
            region.ib_left_pixel
            <= region.peak_pixel_index
            <= region.ib_right_pixel
        )

        # ----------------------------------------------------
        # Save diagnostic figures
        # ----------------------------------------------------

        plot_full_spectrum(
            region=region,
            output_directory=OUTPUT_DIRECTORY,
        )

        plot_peak_region_linear(
            region=region,
            output_directory=OUTPUT_DIRECTORY,
        )

        plot_peak_region_log(
            region=region,
            output_directory=OUTPUT_DIRECTORY,
        )

        plot_stray_light_only(
            region=region,
            output_directory=OUTPUT_DIRECTORY,
        )

    plot_all_stray_light_regions(
        regions=regions,
        output_directory=OUTPUT_DIRECTORY,
    )

    print()
    print(
        "Diagnostic plots saved to:"
    )

    print(
        OUTPUT_DIRECTORY
    )

    print()
    print(
        "Step 5 stray-light region validation passed."
    )


if __name__ == "__main__":
    main()