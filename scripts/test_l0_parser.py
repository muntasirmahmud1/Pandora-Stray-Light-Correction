from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from pandora_straylight.l0_parser import load_pandora_l0


CALIBRATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Pandora63s1_LabGSFC_20250311_L0.txt"
)


def main() -> None:
    l0_file = load_pandora_l0(CALIBRATION_FILE)

    comment_records = [
        record
        for record in l0_file.records
        if record.is_comment
    ]

    measurement_records = [
        record
        for record in l0_file.records
        if not record.is_comment
    ]

    print("=" * 70)
    print("Pandora L0 parser test")
    print("=" * 70)

    print(f"File: {l0_file.file_path}")
    print(
        f"Data begins at file line: "
        f"{l0_file.data_start_line_number}"
    )
    print(f"Header lines: {len(l0_file.header_lines)}")
    print(f"Total parsed records: {len(l0_file.records)}")
    print(f"Comment records: {len(comment_records)}")
    print(f"Measurement records: {len(measurement_records)}")

    if not measurement_records:
        raise RuntimeError(
            "No numerical measurement records were found."
        )

    first_measurement = measurement_records[0]

    print()
    print("First numerical measurement")
    print("-" * 70)
    print(f"File line: {first_measurement.line_number}")
    print(f"Routine code: {first_measurement.routine_code}")
    print(f"Timestamp: {first_measurement.timestamp}")
    print(f"Routine count: {first_measurement.routine_count}")
    print(
        f"Repetition count: "
        f"{first_measurement.repetition_count}"
    )

    print(
        f"Working pixels: "
        f"{first_measurement.working_pixels.size}"
    )

    print(
        f"Blind pixels: "
        f"{first_measurement.blind_pixels.size}"
    )

    if first_measurement.uncertainty_values is None:
        print("Uncertainty values: none")
    else:
        print(
            f"Uncertainty values: "
            f"{first_measurement.uncertainty_values.size}"
        )

    print()
    print("Last four blind-pixel values:")
    print(first_measurement.blind_pixels)

    assert first_measurement.working_pixels.size == 2048
    assert first_measurement.blind_pixels.size == 4

    print()
    print("Step 1 parser validation passed.")


if __name__ == "__main__":
    main()