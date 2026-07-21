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
from pandora_straylight.lsf_processing import (
    build_all_normalized_straylight_distributions,
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
    / "normalized_straylight_distributions"
)


# ============================================================
# Processing settings
# ============================================================

# This must match the value tested in Step 5.
IB_REGION_SIZE = 21


# Logarithmic plotting limits for normalized matrix response.
LOG_Y_MIN = 1e-9
LOG_Y_MAX = 1e-2


# ============================================================
# Plot helpers
# ============================================================

def prepare_positive_log_values(
    values: np.ndarray,
) -> np.ndarray:
    """
    Prepare positive values for logarithmic plotting.

    Zero and negative values are replaced by NaN in the plotting
    copy only.
    """

    plot_values = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    invalid = (
        ~np.isfinite(plot_values)
        | (plot_values <= 0)
    )

    plot_values[invalid] = np.nan

    return plot_values


def plot_one_distribution(
    distribution,
    output_directory: Path,
) -> None:
    """
    Save one normalized stray-light distribution.
    """

    pixels = np.arange(
        distribution.positive_distribution.size
    )

    plot_values = prepare_positive_log_values(
        distribution.positive_distribution
    )

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        pixels,
        plot_values,
        linewidth=1.0,
        label="Positive normalized stray-light response",
    )

    axis.axvline(
        distribution.source_peak_pixel,
        linestyle="--",
        linewidth=1.0,
        label="Source laser pixel",
    )

    axis.axvspan(
        distribution.ib_left_pixel,
        distribution.ib_right_pixel,
        alpha=0.2,
        label="Excluded in-band region",
    )

    axis.set_yscale("log")

    axis.set_xlim(
        0,
        distribution.positive_distribution.size - 1,
    )

    axis.set_ylim(
        LOG_Y_MIN,
        LOG_Y_MAX,
    )

    axis.set_xlabel("Receiving detector pixel")

    axis.set_ylabel(
        "Stray-light response / integrated in-band signal"
    )

    axis.set_title(
        f"{distribution.wavelength_nm:g} nm normalized "
        f"stray-light distribution\n"
        f"Source pixel = {distribution.source_peak_pixel}, "
        f"total positive fraction = "
        f"{100.0 * distribution.total_straylight_fraction:.5f}%"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    filename = (
        f"normalized_straylight_"
        f"{distribution.wavelength_nm:g}nm"
        f"_occ{distribution.wavelength_occurrence}.png"
    )

    figure.savefig(
        output_directory / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_all_distributions(
    distributions,
    output_directory: Path,
) -> None:
    """
    Plot all normalized distributions on one figure.
    """

    pixels = np.arange(2048)

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    for distribution in distributions:
        plot_values = prepare_positive_log_values(
            distribution.positive_distribution
        )

        label = (
            f"{distribution.wavelength_nm:g} nm "
            f"(occ. {distribution.wavelength_occurrence}, "
            f"src px {distribution.source_peak_pixel})"
        )

        axis.plot(
            pixels,
            plot_values,
            linewidth=0.9,
            label=label,
        )

    axis.set_yscale("log")

    axis.set_xlim(
        0,
        2047,
    )

    axis.set_ylim(
        LOG_Y_MIN,
        LOG_Y_MAX,
    )

    axis.set_xlabel("Receiving detector pixel")

    axis.set_ylabel(
        "Stray-light response / integrated in-band signal"
    )

    axis.set_title(
        "Normalized stray-light distributions from all laser blocks"
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

    figure.savefig(
        output_directory
        / "all_normalized_straylight_distributions.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_distributions_by_source_offset(
    distributions,
    output_directory: Path,
) -> None:
    """
    Shift every distribution so that its source laser peak is at
    offset zero.

    This helps determine whether the stray-light response has a
    similar detector-offset shape at different source wavelengths.
    """

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    for distribution in distributions:
        detector_pixels = np.arange(
            distribution.positive_distribution.size
        )

        pixel_offset = (
            detector_pixels
            - distribution.source_peak_pixel
        )

        plot_values = prepare_positive_log_values(
            distribution.positive_distribution
        )

        label = (
            f"{distribution.wavelength_nm:g} nm "
            f"(occ. {distribution.wavelength_occurrence})"
        )

        axis.plot(
            pixel_offset,
            plot_values,
            linewidth=0.9,
            label=label,
        )

    axis.set_yscale("log")

    axis.set_xlim(
        -2047,
        2047,
    )

    axis.set_ylim(
        LOG_Y_MIN,
        LOG_Y_MAX,
    )

    axis.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
        label="Source laser pixel",
    )

    axis.set_xlabel(
        "Receiving pixel − source laser pixel"
    )

    axis.set_ylabel(
        "Stray-light response / integrated in-band signal"
    )

    axis.set_title(
        "Normalized stray-light response versus source-pixel offset"
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

    figure.savefig(
        output_directory
        / "all_normalized_straylight_by_pixel_offset.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def compare_duplicate_488(
    distributions,
) -> None:
    """
    Compare the two normalized 488 nm distributions.
    """

    distributions_488 = [
        distribution
        for distribution in distributions
        if distribution.wavelength_nm == 488.0
    ]

    print()
    print("=" * 100)
    print("Normalized 488 nm distribution comparison")
    print("=" * 100)

    if len(distributions_488) != 2:
        print(
            f"Expected two 488 nm distributions, but found "
            f"{len(distributions_488)}."
        )
        return

    first = distributions_488[0]
    second = distributions_488[1]

    first_positive = first.positive_distribution
    second_positive = second.positive_distribution

    full_correlation = float(
        np.corrcoef(
            first_positive,
            second_positive,
        )[0, 1]
    )

    difference = (
        second_positive
        - first_positive
    )

    rmse = float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )

    mean_absolute_difference = float(
        np.mean(
            np.abs(difference)
        )
    )

    fraction_difference_percent = (
        100.0
        * (
            second.total_straylight_fraction
            - first.total_straylight_fraction
        )
        / first.total_straylight_fraction
    )

    print(
        f"Occurrence 1 total positive fraction: "
        f"{first.total_straylight_fraction:.10e} "
        f"({100.0 * first.total_straylight_fraction:.6f}%)"
    )

    print(
        f"Occurrence 2 total positive fraction: "
        f"{second.total_straylight_fraction:.10e} "
        f"({100.0 * second.total_straylight_fraction:.6f}%)"
    )

    print(
        f"Relative total-fraction difference: "
        f"{fraction_difference_percent:+.4f}%"
    )

    print(
        f"Distribution correlation: "
        f"{full_correlation:.8f}"
    )

    print(
        f"Distribution RMSE: "
        f"{rmse:.10e}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_absolute_difference:.10e}"
    )


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

    distributions = (
        build_all_normalized_straylight_distributions(
            regions
        )
    )

    print("=" * 155)
    print("Normalized stray-light distribution diagnostics")
    print("=" * 155)

    print(
        f"In-band region size: "
        f"{IB_REGION_SIZE} pixels"
    )

    print()

    print(
        f"{'Wave':<8}"
        f"{'Occ':<5}"
        f"{'Source':<8}"
        f"{'IB norm':<15}"
        f"{'Signed sum':<15}"
        f"{'Positive sum':<15}"
        f"{'SL %':<12}"
        f"{'Neg px':<9}"
        f"{'Pos px':<9}"
        f"{'Max px':<9}"
        f"{'Max value':<15}"
        f"{'Left %':<11}"
        f"{'Right %':<11}"
    )

    print("-" * 155)

    for distribution in distributions:
        print(
            f"{distribution.wavelength_nm:<8g}"
            f"{distribution.wavelength_occurrence:<5}"
            f"{distribution.source_peak_pixel:<8}"
            f"{distribution.normalization_value:<15.3f}"
            f"{distribution.signed_sum:<15.8e}"
            f"{distribution.positive_sum:<15.8e}"
            f"{100.0 * distribution.total_straylight_fraction:<11.6f}%"
            f"{distribution.negative_pixel_count:<9}"
            f"{distribution.positive_pixel_count:<9}"
            f"{distribution.maximum_positive_pixel:<9}"
            f"{distribution.maximum_positive_value:<15.8e}"
            f"{100.0 * distribution.left_fraction:<10.4f}%"
            f"{100.0 * distribution.right_fraction:<10.4f}%"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        assert (
            distribution.signed_distribution.size
            == 2048
        )

        assert (
            distribution.positive_distribution.size
            == 2048
        )

        assert np.all(
            np.isfinite(
                distribution.signed_distribution
            )
        )

        assert np.all(
            np.isfinite(
                distribution.positive_distribution
            )
        )

        assert np.all(
            distribution.positive_distribution >= 0
        )

        assert np.all(
            distribution.signed_distribution[
                distribution.ib_left_pixel:
                distribution.ib_right_pixel + 1
            ]
            == 0
        )

        assert np.all(
            distribution.positive_distribution[
                distribution.ib_left_pixel:
                distribution.ib_right_pixel + 1
            ]
            == 0
        )

        assert np.isclose(
            distribution.positive_sum,
            distribution.left_positive_sum
            + distribution.right_positive_sum,
        )

        assert np.isclose(
            distribution.total_straylight_fraction,
            distribution.positive_sum,
        )

        assert (
            0
            <= distribution.source_peak_pixel
            < 2048
        )

        plot_one_distribution(
            distribution=distribution,
            output_directory=OUTPUT_DIRECTORY,
        )

    plot_all_distributions(
        distributions=distributions,
        output_directory=OUTPUT_DIRECTORY,
    )

    plot_distributions_by_source_offset(
        distributions=distributions,
        output_directory=OUTPUT_DIRECTORY,
    )

    compare_duplicate_488(
        distributions
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
        "Step 6 normalized stray-light distribution "
        "validation passed."
    )


if __name__ == "__main__":
    main()