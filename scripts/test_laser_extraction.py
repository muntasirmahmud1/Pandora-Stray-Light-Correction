from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from pandora_straylight.l0_parser import load_pandora_l0
from pandora_straylight.laser_detection import detect_laser_blocks
from pandora_straylight.laser_extraction import (
    extract_all_laser_spectra,
)


CALIBRATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Pandora63s1_LabGSFC_20250311_L0.txt"
)


def compare_duplicate_488(spectra) -> None:
    """
    Compare the two extracted 488 nm measurements.
    """

    spectra_488 = [
        spectrum
        for spectrum in spectra
        if spectrum.wavelength_nm == 488.0
    ]

    print()
    print("=" * 100)
    print("Duplicate 488 nm comparison")
    print("=" * 100)

    if len(spectra_488) != 2:
        print(
            f"Expected two 488 nm spectra, but found "
            f"{len(spectra_488)}."
        )
        return

    first = spectra_488[0]
    second = spectra_488[1]

    first_spectrum = first.corrected_counts_per_cycle
    second_spectrum = second.corrected_counts_per_cycle

    correlation = float(
        np.corrcoef(
            first_spectrum,
            second_spectrum,
        )[0, 1]
    )

    peak_difference_pixels = (
        second.peak_pixel_index
        - first.peak_pixel_index
    )

    peak_ratio = (
        second.peak_value_counts_per_cycle
        / first.peak_value_counts_per_cycle
    )

    relative_peak_difference_percent = (
        100.0
        * (
            second.peak_value_counts_per_cycle
            - first.peak_value_counts_per_cycle
        )
        / first.peak_value_counts_per_cycle
    )

    normalized_first = (
        first_spectrum
        / first.peak_value_counts_per_cycle
    )

    normalized_second = (
        second_spectrum
        / second.peak_value_counts_per_cycle
    )

    normalized_rmse = float(
        np.sqrt(
            np.mean(
                (
                    normalized_first
                    - normalized_second
                )
                ** 2
            )
        )
    )

    print(
        f"Occurrence 1 routine count: "
        f"{first.routine_count}"
    )
    print(
        f"Occurrence 2 routine count: "
        f"{second.routine_count}"
    )

    print(
        f"Occurrence 1 integration time: "
        f"{first.integration_time_ms:g} ms"
    )
    print(
        f"Occurrence 2 integration time: "
        f"{second.integration_time_ms:g} ms"
    )

    print(
        f"Occurrence 1 peak pixel: "
        f"{first.peak_pixel_index}"
    )
    print(
        f"Occurrence 2 peak pixel: "
        f"{second.peak_pixel_index}"
    )

    print(
        f"Peak-pixel difference: "
        f"{peak_difference_pixels:+d} pixels"
    )

    print(
        f"Occurrence 1 corrected peak: "
        f"{first.peak_value_counts_per_cycle:.3f} "
        "counts/cycle"
    )

    print(
        f"Occurrence 2 corrected peak: "
        f"{second.peak_value_counts_per_cycle:.3f} "
        "counts/cycle"
    )

    print(f"Peak ratio occurrence2/occurrence1: {peak_ratio:.6f}")

    print(
        f"Relative peak difference: "
        f"{relative_peak_difference_percent:+.3f}%"
    )

    print(
        f"Full-spectrum correlation: "
        f"{correlation:.8f}"
    )

    print(
        f"Peak-normalized spectrum RMSE: "
        f"{normalized_rmse:.8f}"
    )


def main() -> None:
    l0_file = load_pandora_l0(CALIBRATION_FILE)
    laser_blocks = detect_laser_blocks(l0_file)

    extracted_spectra = extract_all_laser_spectra(
        laser_blocks
    )

    print("=" * 120)
    print("Extracted unsaturated laser spectra")
    print("=" * 120)

    print(
        f"{'Wave':<9}"
        f"{'Occ':<6}"
        f"{'Count':<8}"
        f"{'Bright':<8}"
        f"{'Dark':<8}"
        f"{'Int ms':<11}"
        f"{'B cycles':<10}"
        f"{'D cycles':<10}"
        f"{'Peak px':<10}"
        f"{'Peak/cycle':<16}"
        f"{'Median':<14}"
        f"{'Minimum':<14}"
    )

    print("-" * 120)

    for spectrum in extracted_spectra:
        print(
            f"{spectrum.wavelength_nm:<9g}"
            f"{spectrum.wavelength_occurrence:<6}"
            f"{spectrum.routine_count:<8}"
            f"{spectrum.bright_repetition:<8}"
            f"{spectrum.dark_repetition:<8}"
            f"{spectrum.integration_time_ms:<11.3f}"
            f"{spectrum.bright_cycles:<10}"
            f"{spectrum.dark_cycles:<10}"
            f"{spectrum.peak_pixel_index:<10}"
            f"{spectrum.peak_value_counts_per_cycle:<16.3f}"
            f"{spectrum.median_corrected_value:<14.3f}"
            f"{spectrum.minimum_corrected_value:<14.3f}"
        )

    assert len(extracted_spectra) == len(laser_blocks)

    for spectrum in extracted_spectra:
        assert spectrum.bright_repetition == 3
        assert spectrum.dark_repetition == 4

        assert (
            spectrum.corrected_counts_per_cycle.size
            == 2048
        )

        assert spectrum.bright_saturation_index == 0
        assert spectrum.dark_saturation_index == 0

        assert np.all(
            np.isfinite(
                spectrum.corrected_counts_per_cycle
            )
        )

        assert spectrum.peak_value_counts_per_cycle > 0

    compare_duplicate_488(extracted_spectra)

    print()
    print("Step 4 laser extraction validation passed.")


if __name__ == "__main__":
    main()