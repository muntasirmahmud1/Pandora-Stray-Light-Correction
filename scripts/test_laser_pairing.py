from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from pandora_straylight.l0_parser import load_pandora_l0
from pandora_straylight.laser_detection import detect_laser_blocks
from pandora_straylight.laser_pairing import (
    build_candidate_pairs,
)


CALIBRATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Pandora63s1_LabGSFC_20250311_L0.txt"
)


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def print_measurement_row(
    label: str,
    summary,
) -> None:
    print(
        f"{label:<8}"
        f"{summary.repetition_count:<6}"
        f"{summary.integration_time_ms:<12.3f}"
        f"{summary.number_of_cycles:<9}"
        f"{summary.saturation_index:<10}"
        f"{yes_no(summary.saturated):<11}"
        f"{summary.filterwheel_1_position:<6}"
        f"{summary.filterwheel_2_position:<6}"
        f"{summary.maximum_pixel_index:<10}"
        f"{summary.maximum_count:<14.0f}"
        f"{summary.median_count:<14.0f}"
    )


def main() -> None:
    l0_file = load_pandora_l0(CALIBRATION_FILE)
    laser_blocks = detect_laser_blocks(l0_file)

    print("=" * 120)
    print("Pandora laser bright/dark pairing inspection")
    print("=" * 120)

    for block_index, block in enumerate(laser_blocks, start=1):
        pairs = build_candidate_pairs(block)

        print()
        print("=" * 120)
        print(
            f"Block {block_index}: "
            f"{block.wavelength_nm:g} nm, "
            f"occurrence {block.occurrence_number}, "
            f"routine count {block.routine_count}"
        )
        print("=" * 120)

        print(
            f"{'Type':<8}"
            f"{'Rep':<6}"
            f"{'Int ms':<12}"
            f"{'Cycles':<9}"
            f"{'Sat idx':<10}"
            f"{'Saturated':<11}"
            f"{'FW1':<6}"
            f"{'FW2':<6}"
            f"{'Peak px':<10}"
            f"{'Maximum':<14}"
            f"{'Median':<14}"
        )

        print("-" * 120)

        for pair in pairs:
            print_measurement_row(
                "Bright",
                pair.bright_summary,
            )

            print_measurement_row(
                "Dark",
                pair.dark_summary,
            )

            print(
                f"Pair {pair.pair_number}: "
                f"integration times match = "
                f"{yes_no(pair.integration_times_match)}, "
                f"filter positions differ = "
                f"{yes_no(pair.filter_positions_differ)}"
            )

            print("-" * 120)

        if len(pairs) != 3:
            print(
                f"WARNING: Expected 3 pairs but found "
                f"{len(pairs)}."
            )

    print()
    print("=" * 120)
    print("488 nm comparison")
    print("=" * 120)

    blocks_488 = [
        block
        for block in laser_blocks
        if block.wavelength_nm == 488.0
    ]

    for block in blocks_488:
        pairs = build_candidate_pairs(block)

        print()
        print(
            f"488 nm occurrence {block.occurrence_number}, "
            f"routine count {block.routine_count}"
        )

        for pair in pairs:
            bright = pair.bright_summary
            dark = pair.dark_summary

            dark_subtracted_peak = (
                bright.maximum_count
                - dark.median_count
            )

            print(
                f"Pair {pair.pair_number}: "
                f"bright rep={bright.repetition_count}, "
                f"dark rep={dark.repetition_count}, "
                f"int={bright.integration_time_ms:g} ms, "
                f"sat_index={bright.saturation_index}, "
                f"saturated={yes_no(bright.saturated)}, "
                f"peak_pixel={bright.maximum_pixel_index}, "
                f"bright_max={bright.maximum_count:.0f}, "
                f"dark_median={dark.median_count:.0f}, "
                f"approx_net_peak={dark_subtracted_peak:.0f}"
            )

    print()
    print("Step 3 bright/dark inspection completed.")


if __name__ == "__main__":
    main()