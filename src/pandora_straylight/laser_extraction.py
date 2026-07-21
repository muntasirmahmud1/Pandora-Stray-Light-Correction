from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from pandora_straylight.l0_parser import PandoraL0Record
from pandora_straylight.laser_detection import LaserBlock
from pandora_straylight.laser_pairing import (
    LaserBrightDarkPair,
    build_candidate_pairs,
)


@dataclass
class ExtractedLaserSpectrum:
    """
    One selected and dark-corrected laser spectrum.

    Duplicate wavelengths remain separate. For example, two 488 nm
    measurements will produce two ExtractedLaserSpectrum objects.
    """

    wavelength_nm: float
    wavelength_occurrence: int

    routine_code: str
    routine_count: int

    bright_repetition: int
    dark_repetition: int

    bright_line_number: int
    dark_line_number: int

    integration_time_ms: float
    bright_cycles: int
    dark_cycles: int

    bright_saturation_index: int
    dark_saturation_index: int

    bright_counts_per_cycle: np.ndarray
    dark_counts_per_cycle: np.ndarray
    corrected_counts_per_cycle: np.ndarray

    peak_pixel_index: int
    peak_value_counts_per_cycle: float

    minimum_corrected_value: float
    median_corrected_value: float


def validate_selected_pair(
    block: LaserBlock,
    pair: LaserBrightDarkPair,
) -> None:
    """
    Validate a bright/dark pair before extracting its spectrum.
    """

    bright = pair.bright_record
    dark = pair.dark_record

    if bright.working_pixels is None:
        raise ValueError(
            f"Bright measurement on line {bright.line_number} "
            "does not contain working pixels."
        )

    if dark.working_pixels is None:
        raise ValueError(
            f"Dark measurement on line {dark.line_number} "
            "does not contain working pixels."
        )

    if bright.working_pixels.size != 2048:
        raise ValueError(
            f"Bright measurement on line {bright.line_number} "
            f"contains {bright.working_pixels.size} working pixels "
            "instead of 2048."
        )

    if dark.working_pixels.size != 2048:
        raise ValueError(
            f"Dark measurement on line {dark.line_number} "
            f"contains {dark.working_pixels.size} working pixels "
            "instead of 2048."
        )

    if not pair.integration_times_match:
        raise ValueError(
            f"Bright and dark integration times do not match for "
            f"{block.wavelength_nm:g} nm, routine count "
            f"{block.routine_count}."
        )

    if bright.number_of_cycles is None:
        raise ValueError(
            f"Bright measurement on line {bright.line_number} "
            "does not contain a valid cycle count."
        )

    if dark.number_of_cycles is None:
        raise ValueError(
            f"Dark measurement on line {dark.line_number} "
            "does not contain a valid cycle count."
        )

    if bright.number_of_cycles <= 0:
        raise ValueError(
            f"Bright measurement on line {bright.line_number} "
            "has an invalid cycle count."
        )

    if dark.number_of_cycles <= 0:
        raise ValueError(
            f"Dark measurement on line {dark.line_number} "
            "has an invalid cycle count."
        )

    if pair.bright_summary.saturated:
        raise ValueError(
            f"Selected bright measurement for "
            f"{block.wavelength_nm:g} nm is saturated."
        )

    if pair.dark_summary.saturated:
        raise ValueError(
            f"Selected dark measurement for "
            f"{block.wavelength_nm:g} nm is saturated."
        )


def select_first_unsaturated_pair(
    block: LaserBlock,
) -> LaserBrightDarkPair:
    """
    Select the first valid unsaturated bright/dark pair.

    Candidate pairs are checked in their original order:

        pair 1: repetitions 3 and 4
        pair 2: repetitions 5 and 6
        pair 3: repetitions 7 and 8

    The first pair satisfying all conditions is returned.
    """

    candidate_pairs = build_candidate_pairs(block)

    if not candidate_pairs:
        raise ValueError(
            f"No bright/dark pairs were found for "
            f"{block.wavelength_nm:g} nm, routine count "
            f"{block.routine_count}."
        )

    for pair in candidate_pairs:
        bright_is_valid = not pair.bright_summary.saturated
        dark_is_valid = not pair.dark_summary.saturated

        cycles_are_valid = (
            pair.bright_record.number_of_cycles is not None
            and pair.bright_record.number_of_cycles > 0
            and pair.dark_record.number_of_cycles is not None
            and pair.dark_record.number_of_cycles > 0
        )

        if (
            bright_is_valid
            and dark_is_valid
            and pair.integration_times_match
            and cycles_are_valid
        ):
            return pair

    raise ValueError(
        f"No valid unsaturated bright/dark pair was found for "
        f"{block.wavelength_nm:g} nm, routine count "
        f"{block.routine_count}."
    )


def calculate_counts_per_cycle(
    record: PandoraL0Record,
) -> np.ndarray:
    """
    Convert summed detector counts to counts per cycle.
    """

    if record.working_pixels is None:
        raise ValueError(
            f"Measurement on line {record.line_number} "
            "does not contain working pixels."
        )

    if (
        record.number_of_cycles is None
        or record.number_of_cycles <= 0
    ):
        raise ValueError(
            f"Measurement on line {record.line_number} "
            "does not contain a valid cycle count."
        )

    return (
        np.asarray(record.working_pixels, dtype=np.float64)
        / float(record.number_of_cycles)
    )


def extract_laser_spectrum(
    block: LaserBlock,
) -> ExtractedLaserSpectrum:
    """
    Select the first unsaturated pair and calculate its
    cycle-corrected, dark-subtracted spectrum.
    """

    selected_pair = select_first_unsaturated_pair(block)

    validate_selected_pair(
        block=block,
        pair=selected_pair,
    )

    bright = selected_pair.bright_record
    dark = selected_pair.dark_record

    bright_counts_per_cycle = calculate_counts_per_cycle(bright)
    dark_counts_per_cycle = calculate_counts_per_cycle(dark)

    corrected_counts_per_cycle = (
        bright_counts_per_cycle
        - dark_counts_per_cycle
    )

    peak_pixel_index = int(
        np.argmax(corrected_counts_per_cycle)
    )

    peak_value_counts_per_cycle = float(
        corrected_counts_per_cycle[peak_pixel_index]
    )

    return ExtractedLaserSpectrum(
        wavelength_nm=block.wavelength_nm,
        wavelength_occurrence=block.occurrence_number,
        routine_code=block.routine_code,
        routine_count=block.routine_count,
        bright_repetition=bright.repetition_count,
        dark_repetition=dark.repetition_count,
        bright_line_number=bright.line_number,
        dark_line_number=dark.line_number,
        integration_time_ms=float(bright.integration_time_ms),
        bright_cycles=int(bright.number_of_cycles),
        dark_cycles=int(dark.number_of_cycles),
        bright_saturation_index=int(
            bright.saturation_index or 0
        ),
        dark_saturation_index=int(
            dark.saturation_index or 0
        ),
        bright_counts_per_cycle=bright_counts_per_cycle,
        dark_counts_per_cycle=dark_counts_per_cycle,
        corrected_counts_per_cycle=corrected_counts_per_cycle,
        peak_pixel_index=peak_pixel_index,
        peak_value_counts_per_cycle=peak_value_counts_per_cycle,
        minimum_corrected_value=float(
            np.min(corrected_counts_per_cycle)
        ),
        median_corrected_value=float(
            np.median(corrected_counts_per_cycle)
        ),
    )


def extract_all_laser_spectra(
    laser_blocks: List[LaserBlock],
) -> List[ExtractedLaserSpectrum]:
    """
    Extract one valid unsaturated spectrum from every laser block.

    Duplicate wavelengths remain separate.
    """

    extracted_spectra: List[ExtractedLaserSpectrum] = []

    for block in laser_blocks:
        extracted_spectra.append(
            extract_laser_spectrum(block)
        )

    return extracted_spectra