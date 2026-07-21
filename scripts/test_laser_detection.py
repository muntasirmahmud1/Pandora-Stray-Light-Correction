from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from pandora_straylight.l0_parser import load_pandora_l0
from pandora_straylight.laser_detection import detect_laser_blocks


CALIBRATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Pandora63s1_LabGSFC_20250311_L0.txt"
)


def main() -> None:
    l0_file = load_pandora_l0(CALIBRATION_FILE)

    laser_blocks = detect_laser_blocks(l0_file)

    print("=" * 85)
    print("Pandora laser-block detection test")
    print("=" * 85)

    print(f"File: {l0_file.file_path}")
    print(f"Detected laser blocks: {len(laser_blocks)}")
    print()

    for block_index, block in enumerate(laser_blocks, start=1):
        repetition_numbers = [
            record.repetition_count
            for record in block.measurement_records
        ]

        timestamps = [
            record.timestamp
            for record in block.measurement_records
        ]

        print("-" * 85)
        print(f"Laser block: {block_index}")
        print(f"Wavelength: {block.wavelength_nm:g} nm")
        print(
            f"Wavelength occurrence: "
            f"{block.occurrence_number}"
        )
        print(f"Routine code: {block.routine_code}")
        print(f"Routine count: {block.routine_count}")
        print(f"Marker file line: {block.marker_line_number}")
        print(
            f"Numerical measurements: "
            f"{len(block.measurement_records)}"
        )
        print(f"Repetition numbers: {repetition_numbers}")
        print(f"Timestamps: {timestamps}")

    print()
    print("=" * 85)
    print("Compact summary")
    print("=" * 85)

    print(
        f"{'Block':<8}"
        f"{'Wavelength':<16}"
        f"{'Occurrence':<14}"
        f"{'Routine':<10}"
        f"{'Count':<10}"
        f"{'Measurements':<15}"
        f"{'Repetitions'}"
    )

    for block_index, block in enumerate(laser_blocks, start=1):
        repetitions = [
            record.repetition_count
            for record in block.measurement_records
        ]

        print(
            f"{block_index:<8}"
            f"{block.wavelength_nm:<16g}"
            f"{block.occurrence_number:<14}"
            f"{block.routine_code:<10}"
            f"{block.routine_count:<10}"
            f"{len(block.measurement_records):<15}"
            f"{repetitions}"
        )

    wavelengths = [
        block.wavelength_nm
        for block in laser_blocks
    ]

    duplicate_488_blocks = [
        block
        for block in laser_blocks
        if block.wavelength_nm == 488.0
    ]

    print()
    print(f"Detected wavelengths: {wavelengths}")
    print(
        f"Number of separate 488 nm blocks: "
        f"{len(duplicate_488_blocks)}"
    )

    assert laser_blocks, "No laser blocks were detected."

    assert len(duplicate_488_blocks) == 2, (
        "Expected two separate 488 nm laser blocks."
    )

    for block in laser_blocks:
        if not block.measurement_records:
            raise AssertionError(
                f"The {block.wavelength_nm:g} nm laser block "
                f"with routine count {block.routine_count} "
                "contains no numerical measurements."
            )

        for measurement in block.measurement_records:
            assert (
                measurement.routine_count
                == block.routine_count
            )

            assert (
                measurement.routine_code
                == block.routine_code
            )

    print()
    print("Step 2 laser-block detection validation passed.")


if __name__ == "__main__":
    main()