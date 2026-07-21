from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


SEPARATOR_PREFIX = "-----"

# Pandora L0 structure:
# Columns 1-24: metadata
# Columns 25-2076: 2052 detector values
# Last 4 detector values are blind pixels
METADATA_COLUMN_COUNT = 24
TOTAL_DETECTOR_PIXEL_COUNT = 2052
WORKING_PIXEL_COUNT = 2048
BLIND_PIXEL_COUNT = 4


@dataclass
class PandoraL0Record:
    """
    One parsed line from a Pandora L0 file.

    A record may be:
    - a comment line, such as an INFO Laser line
    - a numerical measurement line
    """

    original_line: str
    line_number: int
    routine_code: str
    timestamp: Optional[str]
    routine_count: Optional[int]
    repetition_count: Optional[int]
    is_comment: bool
    
    duration_seconds: Optional[float] = None
    integration_time_ms: Optional[float] = None
    number_of_cycles: Optional[int] = None
    saturation_index: Optional[int] = None
    filterwheel_1_position: Optional[int] = None
    filterwheel_2_position: Optional[int] = None
    scale_factor: Optional[float] = None
    uncertainty_indicator: Optional[int] = None
    
    metadata_values: Optional[List[str]] = None
    working_pixels: Optional[np.ndarray] = None
    blind_pixels: Optional[np.ndarray] = None
    uncertainty_values: Optional[np.ndarray] = None


@dataclass
class PandoraL0File:
    """
    Complete parsed Pandora L0 file.
    """

    file_path: Path
    header_lines: List[str]
    records: List[PandoraL0Record]
    data_start_line_number: int


def find_data_start(lines: List[str]) -> int:
    """
    Find the line immediately after the second dashed separator.

    Returns
    -------
    int
        Zero-based index of the first record line.
    """

    separator_indices = []

    for index, line in enumerate(lines):
        if line.strip().startswith(SEPARATOR_PREFIX):
            separator_indices.append(index)

    if len(separator_indices) < 2:
        raise ValueError(
            "The Pandora L0 file does not contain at least two "
            "separator lines."
        )

    second_separator_index = separator_indices[1]

    return second_separator_index + 1


def is_comment_line(parts: List[str]) -> bool:
    """
    Determine whether a parsed L0 line is a comment line.

    Pandora comment lines contain '#' in column 5.
    """

    if len(parts) < 5:
        return True

    return parts[4] == "#"


def parse_optional_int(value: str) -> Optional[int]:
    """
    Convert a string to int when possible.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def parse_optional_float(value: str) -> Optional[float]:
    """
    Convert a string to float when possible.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return None
    
def parse_comment_record(
    line: str,
    line_number: int,
    parts: List[str],
) -> PandoraL0Record:
    """
    Parse a Pandora comment or INFO line.
    """

    routine_code = parts[0] if len(parts) >= 1 else ""
    timestamp = parts[1] if len(parts) >= 2 else None

    routine_count = (
        parse_optional_int(parts[2])
        if len(parts) >= 3
        else None
    )

    repetition_count = (
        parse_optional_int(parts[3])
        if len(parts) >= 4
        else None
    )

    return PandoraL0Record(
        original_line=line,
        line_number=line_number,
        routine_code=routine_code,
        timestamp=timestamp,
        routine_count=routine_count,
        repetition_count=repetition_count,
        is_comment=True,
    )


def parse_measurement_record(
    line: str,
    line_number: int,
    parts: List[str],
) -> PandoraL0Record:
    """
    Parse one numerical Pandora L0 measurement line.
    """

    minimum_expected_columns = (
        METADATA_COLUMN_COUNT
        + TOTAL_DETECTOR_PIXEL_COUNT
    )

    if len(parts) < minimum_expected_columns:
        raise ValueError(
            f"Line {line_number} contains only {len(parts)} columns. "
            f"At least {minimum_expected_columns} columns were expected."
        )

    routine_code = parts[0]
    timestamp = parts[1]
    routine_count = parse_optional_int(parts[2])
    repetition_count = parse_optional_int(parts[3])

    metadata_values = parts[:METADATA_COLUMN_COUNT]

    # Pandora column numbers are one-based in the L0 header.
    # Python list indices are zero-based.
    duration_seconds = parse_optional_float(metadata_values[4])
    integration_time_ms = parse_optional_float(metadata_values[5])
    number_of_cycles = parse_optional_int(metadata_values[6])
    saturation_index = parse_optional_int(metadata_values[7])
    filterwheel_1_position = parse_optional_int(metadata_values[8])
    filterwheel_2_position = parse_optional_int(metadata_values[9])
    scale_factor = parse_optional_float(metadata_values[22])
    uncertainty_indicator = parse_optional_int(metadata_values[23])

    detector_start = METADATA_COLUMN_COUNT
    detector_end = detector_start + TOTAL_DETECTOR_PIXEL_COUNT

    detector_values = np.asarray(
        parts[detector_start:detector_end],
        dtype=np.float64,
    )

    if detector_values.size != TOTAL_DETECTOR_PIXEL_COUNT:
        raise ValueError(
            f"Line {line_number} contains "
            f"{detector_values.size} detector values instead of "
            f"{TOTAL_DETECTOR_PIXEL_COUNT}."
        )

    working_pixels = detector_values[:WORKING_PIXEL_COUNT]
    blind_pixels = detector_values[WORKING_PIXEL_COUNT:]

    if blind_pixels.size != BLIND_PIXEL_COUNT:
        raise ValueError(
            f"Line {line_number} contains "
            f"{blind_pixels.size} blind pixels instead of "
            f"{BLIND_PIXEL_COUNT}."
        )

    uncertainty_parts = parts[detector_end:]

    uncertainty_values = None

    if uncertainty_parts:
        uncertainty_values = np.asarray(
            uncertainty_parts,
            dtype=np.float64,
        )

    return PandoraL0Record(
        original_line=line,
        line_number=line_number,
        routine_code=routine_code,
        timestamp=timestamp,
        routine_count=routine_count,
        repetition_count=repetition_count,
        is_comment=False,
        duration_seconds=duration_seconds,
        integration_time_ms=integration_time_ms,
        number_of_cycles=number_of_cycles,
        saturation_index=saturation_index,
        filterwheel_1_position=filterwheel_1_position,
        filterwheel_2_position=filterwheel_2_position,
        scale_factor=scale_factor,
        uncertainty_indicator=uncertainty_indicator,
        metadata_values=metadata_values,
        working_pixels=working_pixels,
        blind_pixels=blind_pixels,
        uncertainty_values=uncertainty_values,
    )


def load_pandora_l0(file_path: str | Path) -> PandoraL0File:
    """
    Read and parse a Pandora L0 file.

    Parameters
    ----------
    file_path
        Path to the Pandora L0 text file.

    Returns
    -------
    PandoraL0File
        Parsed header and records.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Pandora L0 file was not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        lines = file.readlines()

    data_start_index = find_data_start(lines)

    header_lines = lines[:data_start_index]
    record_lines = lines[data_start_index:]

    records: List[PandoraL0Record] = []

    for zero_based_index, line in enumerate(
        record_lines,
        start=data_start_index,
    ):
        stripped_line = line.strip()

        if not stripped_line:
            continue

        line_number = zero_based_index + 1
        parts = stripped_line.split()

        if is_comment_line(parts):
            record = parse_comment_record(
                line=stripped_line,
                line_number=line_number,
                parts=parts,
            )
        else:
            record = parse_measurement_record(
                line=stripped_line,
                line_number=line_number,
                parts=parts,
            )

        records.append(record)

    return PandoraL0File(
        file_path=file_path,
        header_lines=header_lines,
        records=records,
        data_start_line_number=data_start_index + 1,
    )