from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from pandora_straylight.l0_parser import (
    PandoraL0File,
    PandoraL0Record,
)


LASER_MARKER_PATTERN = re.compile(
    r"INFO\s+Laser\s*:\s*.*?"
    r"(?P<wavelength>\d+(?:\.\d+)?)\s*nm",
    flags=re.IGNORECASE,
)


@dataclass
class LaserBlock:
    """
    One laser calibration block detected in a Pandora L0 file.

    Duplicate wavelengths are preserved as separate LaserBlock objects.
    """

    wavelength_nm: float
    routine_code: str
    routine_count: int
    marker_line_number: int
    marker_text: str
    occurrence_number: int
    measurement_records: List[PandoraL0Record]


def is_laser_marker(record: PandoraL0Record) -> bool:
    """
    Return True when a comment record contains an INFO Laser marker.
    """

    if not record.is_comment:
        return False

    return LASER_MARKER_PATTERN.search(record.original_line) is not None


def extract_laser_wavelength(record: PandoraL0Record) -> float:
    """
    Extract the nominal laser wavelength from an INFO Laser line.

    Examples
    --------
    'INFO Laser: Cobolt Laser 405nm' -> 405.0
    'INFO Laser: Argon Laser 488 nm' -> 488.0
    """

    match = LASER_MARKER_PATTERN.search(record.original_line)

    if match is None:
        raise ValueError(
            f"Could not extract a laser wavelength from file line "
            f"{record.line_number}:\n{record.original_line}"
        )

    return float(match.group("wavelength"))


def collect_measurements_for_laser(
    records: List[PandoraL0Record],
    marker_index: int,
) -> List[PandoraL0Record]:
    """
    Collect numerical measurements belonging to one laser marker.

    Measurements are assigned using:
    - the same routine code,
    - the same routine count,
    - their position after the laser marker.

    Collection stops when:
    - another INFO Laser marker is reached, or
    - a record with a different routine count is reached after the
      current routine has begun.
    """

    marker_record = records[marker_index]

    if marker_record.routine_count is None:
        raise ValueError(
            f"Laser marker on line {marker_record.line_number} "
            "does not contain a valid routine count."
        )

    target_routine_code = marker_record.routine_code
    target_routine_count = marker_record.routine_count

    measurements: List[PandoraL0Record] = []
    routine_has_started = False

    for record in records[marker_index + 1:]:
        if is_laser_marker(record):
            break

        same_routine = (
            record.routine_code == target_routine_code
            and record.routine_count == target_routine_count
        )

        if same_routine:
            routine_has_started = True

            if not record.is_comment:
                measurements.append(record)

            continue

        if routine_has_started:
            # Once the requested routine has started, a different
            # routine count indicates the end of this laser block.
            if record.routine_count != target_routine_count:
                break

    return measurements


def detect_laser_blocks(
    l0_file: PandoraL0File,
) -> List[LaserBlock]:
    """
    Detect all laser calibration blocks in a parsed Pandora L0 file.

    Duplicate wavelengths are deliberately retained as separate blocks.

    Parameters
    ----------
    l0_file
        Parsed Pandora L0 file returned by load_pandora_l0().

    Returns
    -------
    list of LaserBlock
        Laser blocks in their original file order.
    """

    laser_blocks: List[LaserBlock] = []
    wavelength_occurrences: dict[float, int] = {}

    for marker_index, record in enumerate(l0_file.records):
        if not is_laser_marker(record):
            continue

        if record.routine_count is None:
            raise ValueError(
                f"Laser marker on line {record.line_number} "
                "has no valid routine count."
            )

        wavelength_nm = extract_laser_wavelength(record)

        wavelength_occurrences[wavelength_nm] = (
            wavelength_occurrences.get(wavelength_nm, 0) + 1
        )

        occurrence_number = wavelength_occurrences[wavelength_nm]

        measurements = collect_measurements_for_laser(
            records=l0_file.records,
            marker_index=marker_index,
        )

        laser_block = LaserBlock(
            wavelength_nm=wavelength_nm,
            routine_code=record.routine_code,
            routine_count=record.routine_count,
            marker_line_number=record.line_number,
            marker_text=record.original_line,
            occurrence_number=occurrence_number,
            measurement_records=measurements,
        )

        laser_blocks.append(laser_block)

    return laser_blocks
