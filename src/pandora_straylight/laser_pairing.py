from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pandora_straylight.l0_parser import PandoraL0Record
from pandora_straylight.laser_detection import LaserBlock


BRIGHT_REPETITIONS = {3, 5, 7}
DARK_REPETITIONS = {4, 6, 8}

EXPECTED_PAIRS = (
    (3, 4),
    (5, 6),
    (7, 8),
)


@dataclass
class LaserMeasurementSummary:
    """
    Diagnostic information for one laser measurement.
    """

    measurement_type: str
    repetition_count: int
    file_line_number: int
    timestamp: Optional[str]

    integration_time_ms: Optional[float]
    number_of_cycles: Optional[int]
    saturation_index: Optional[int]

    filterwheel_1_position: Optional[int]
    filterwheel_2_position: Optional[int]

    minimum_count: float
    median_count: float
    mean_count: float
    maximum_count: float
    maximum_pixel_index: int

    saturated: bool


@dataclass
class LaserBrightDarkPair:
    """
    One candidate bright/dark pair.
    """

    pair_number: int
    bright_record: PandoraL0Record
    dark_record: PandoraL0Record

    integration_times_match: bool
    filter_positions_differ: bool

    bright_summary: LaserMeasurementSummary
    dark_summary: LaserMeasurementSummary


def classify_measurement_type(
    record: PandoraL0Record,
) -> str:
    """
    Classify a laser measurement using its repetition number.

    The observed L5 sequence is:

        repetition 3: bright
        repetition 4: dark
        repetition 5: bright
        repetition 6: dark
        repetition 7: bright
        repetition 8: dark
    """

    repetition = record.repetition_count

    if repetition in BRIGHT_REPETITIONS:
        return "bright"

    if repetition in DARK_REPETITIONS:
        return "dark"

    return "unknown"


def is_saturated(
    record: PandoraL0Record,
) -> bool:
    """
    Determine saturation from Pandora's saturation index.

    Column 8 definition:

    positive:
        Number of saturated cycles included in the result.

    negative:
        Number of cycles skipped because of saturation.

    zero:
        No saturation reported.

    Therefore, any nonzero value indicates that saturation occurred.
    """

    saturation_index = record.saturation_index

    if saturation_index is None:
        return False

    return saturation_index != 0


def summarize_measurement(
    record: PandoraL0Record,
) -> LaserMeasurementSummary:
    """
    Calculate basic diagnostics for one numerical measurement.
    """

    if record.working_pixels is None:
        raise ValueError(
            f"Measurement on line {record.line_number} "
            "does not contain working-pixel data."
        )

    spectrum = np.asarray(
        record.working_pixels,
        dtype=np.float64,
    )

    if spectrum.size != 2048:
        raise ValueError(
            f"Expected 2048 working pixels on line "
            f"{record.line_number}, but found {spectrum.size}."
        )

    return LaserMeasurementSummary(
        measurement_type=classify_measurement_type(record),
        repetition_count=record.repetition_count,
        file_line_number=record.line_number,
        timestamp=record.timestamp,
        integration_time_ms=record.integration_time_ms,
        number_of_cycles=record.number_of_cycles,
        saturation_index=record.saturation_index,
        filterwheel_1_position=record.filterwheel_1_position,
        filterwheel_2_position=record.filterwheel_2_position,
        minimum_count=float(np.min(spectrum)),
        median_count=float(np.median(spectrum)),
        mean_count=float(np.mean(spectrum)),
        maximum_count=float(np.max(spectrum)),
        maximum_pixel_index=int(np.argmax(spectrum)),
        saturated=is_saturated(record),
    )


def find_record_by_repetition(
    block: LaserBlock,
    repetition_count: int,
) -> Optional[PandoraL0Record]:
    """
    Find one numerical measurement by repetition number.
    """

    matches = [
        record
        for record in block.measurement_records
        if record.repetition_count == repetition_count
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Laser {block.wavelength_nm:g} nm, routine count "
            f"{block.routine_count}, contains multiple numerical "
            f"records for repetition {repetition_count}."
        )

    return matches[0]


def integration_times_match(
    bright_record: PandoraL0Record,
    dark_record: PandoraL0Record,
    tolerance_ms: float = 1e-9,
) -> bool:
    """
    Check whether bright and dark integration times match.
    """

    bright_time = bright_record.integration_time_ms
    dark_time = dark_record.integration_time_ms

    if bright_time is None or dark_time is None:
        return False

    return abs(bright_time - dark_time) <= tolerance_ms


def build_candidate_pairs(
    block: LaserBlock,
) -> List[LaserBrightDarkPair]:
    """
    Build the three expected candidate bright/dark pairs.

    Expected pairs:

        3 and 4
        5 and 6
        7 and 8
    """

    pairs: List[LaserBrightDarkPair] = []

    for pair_number, (
        bright_repetition,
        dark_repetition,
    ) in enumerate(EXPECTED_PAIRS, start=1):

        bright_record = find_record_by_repetition(
            block,
            bright_repetition,
        )

        dark_record = find_record_by_repetition(
            block,
            dark_repetition,
        )

        if bright_record is None or dark_record is None:
            continue

        bright_summary = summarize_measurement(bright_record)
        dark_summary = summarize_measurement(dark_record)

        filter_positions_differ = (
            bright_record.filterwheel_1_position
            != dark_record.filterwheel_1_position
            or
            bright_record.filterwheel_2_position
            != dark_record.filterwheel_2_position
        )

        pairs.append(
            LaserBrightDarkPair(
                pair_number=pair_number,
                bright_record=bright_record,
                dark_record=dark_record,
                integration_times_match=integration_times_match(
                    bright_record,
                    dark_record,
                ),
                filter_positions_differ=filter_positions_differ,
                bright_summary=bright_summary,
                dark_summary=dark_summary,
            )
        )

    return pairs