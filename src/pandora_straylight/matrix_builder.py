from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np

from pandora_straylight.laser_extraction import (
    ExtractedLaserSpectrum,
)


DuplicateSourcePolicy = Literal[
    "first",
    "last",
]


@dataclass
class PreparedLaserLSF:
    """
    One laser LSF prepared according to Pandora63_analysis.ipynb.
    """

    wavelength_nm: float
    wavelength_occurrence: int
    routine_count: int

    source_pixel: int

    corrected_spectrum: np.ndarray
    normalized_lsf: np.ndarray
    normalized_sdf_column: np.ndarray

    ib_start: int
    ib_end_exclusive: int
    ib_sum: float


@dataclass
class OriginalNotebookMatrixResult:
    """
    Stray-light matrices constructed using the exact method from
    Pandora63_analysis.ipynb.
    """

    sdf_matrix: np.ndarray
    identity_matrix: np.ndarray
    a_matrix: np.ndarray
    correction_matrix: np.ndarray

    prepared_lsfs: List[PreparedLaserLSF]

    source_pixels: np.ndarray
    wavelengths_nm: np.ndarray

    detector_pixel_count: int
    ib_region_size: int


def normalize_lsf_min_max(
    spectrum: np.ndarray,
) -> np.ndarray:
    """
    Reproduce the notebook normalization:

        (data - min(data)) / (max(data) - min(data))
    """

    spectrum = np.asarray(
        spectrum,
        dtype=np.float64,
    )

    if spectrum.ndim != 1:
        raise ValueError(
            "The laser spectrum must be one-dimensional."
        )

    if not np.all(np.isfinite(spectrum)):
        raise ValueError(
            "The laser spectrum contains non-finite values."
        )

    minimum = float(
        np.min(spectrum)
    )

    maximum = float(
        np.max(spectrum)
    )

    denominator = maximum - minimum

    if denominator <= 0:
        raise ValueError(
            "Cannot normalize a constant laser spectrum."
        )

    return (
        spectrum - minimum
    ) / denominator


def apply_original_325nm_repairs(
    spectrum: np.ndarray,
) -> np.ndarray:
    """
    Apply the manual 325 nm replacements found in
    Pandora63_analysis.ipynb.

    These are reproduced exactly from the original notebook:

        700:800   <- 600:700
        800:900   <- 600:700
        1110:1210 <- 1000:1100
        1350:1400 <- 1000:1050
        1430:1480 <- 1000:1050
        1530:1580 <- 1000:1050
        1600:1700 <- 1000:1100
        1860:1960 <- 1000:1100
    """

    repaired = np.asarray(
        spectrum,
        dtype=np.float64,
    ).copy()

    if repaired.size != 2048:
        raise ValueError(
            "The 325 nm repair expects 2048 detector pixels."
        )

    repaired[700:800] = repaired[600:700]
    repaired[800:900] = repaired[600:700]
    repaired[1110:1210] = repaired[1000:1100]
    repaired[1350:1400] = repaired[1000:1050]
    repaired[1430:1480] = repaired[1000:1050]
    repaired[1530:1580] = repaired[1000:1050]
    repaired[1600:1700] = repaired[1000:1100]
    repaired[1860:1960] = repaired[1000:1100]

    return repaired


def normalize_lsf_correct(
    normalized_lsf: np.ndarray,
    pixel_number: int,
    ib_region_size: int,
) -> tuple[
    np.ndarray,
    int,
    int,
    float,
]:
    """
    Reproduce normalize_lsf_correct() from the original notebook.

    Important
    ---------
    For ib_region_size=20, this function removes 21 total pixels:

        source_pixel - 10
        through
        source_pixel + 10
    """

    lsf = np.asarray(
        normalized_lsf,
        dtype=np.float64,
    ).copy()

    if lsf.ndim != 1:
        raise ValueError(
            "The normalized LSF must be one-dimensional."
        )

    if ib_region_size <= 0:
        raise ValueError(
            "ib_region_size must be positive."
        )

    if not (
        0 <= pixel_number < lsf.size
    ):
        raise ValueError(
            f"Source pixel {pixel_number} is outside the detector."
        )

    ib_start = max(
        0,
        pixel_number - ib_region_size // 2,
    )

    ib_end = min(
        len(lsf),
        pixel_number + ib_region_size // 2 + 1,
    )

    ib_region = np.arange(
        ib_start,
        ib_end,
    )

    ib_sum = float(
        np.sum(
            lsf[ib_region]
        )
    )

    if not np.isfinite(ib_sum):
        raise ValueError(
            "The in-band LSF sum is not finite."
        )

    if ib_sum == 0:
        raise ValueError(
            "The in-band LSF sum is zero."
        )

    lsf[ib_region] = 0.0

    normalized_sdf_column = (
        lsf / ib_sum
    )

    return (
        normalized_sdf_column,
        ib_start,
        ib_end,
        ib_sum,
    )


def select_unique_source_spectra(
    extracted_spectra: List[
        ExtractedLaserSpectrum
    ],
    duplicate_source_policy: DuplicateSourcePolicy = "first",
) -> List[ExtractedLaserSpectrum]:
    """
    Select one spectrum for every unique source pixel.

    The original notebook contains one 488 nm LSF. Therefore, when
    automatic extraction finds duplicate 488 nm blocks at the same
    source pixel, the first occurrence is selected by default.
    """

    if duplicate_source_policy not in {
        "first",
        "last",
    }:
        raise ValueError(
            "duplicate_source_policy must be 'first' or 'last'."
        )

    if not extracted_spectra:
        raise ValueError(
            "No extracted laser spectra were supplied."
        )

    sorted_spectra = sorted(
        extracted_spectra,
        key=lambda item: (
            int(item.peak_pixel_index),
            int(item.wavelength_occurrence),
        ),
    )

    selected_by_pixel: dict[
        int,
        ExtractedLaserSpectrum,
    ] = {}

    for spectrum in sorted_spectra:
        source_pixel = int(
            spectrum.peak_pixel_index
        )

        if duplicate_source_policy == "first":
            selected_by_pixel.setdefault(
                source_pixel,
                spectrum,
            )
        else:
            selected_by_pixel[
                source_pixel
            ] = spectrum

    selected = [
        selected_by_pixel[source_pixel]
        for source_pixel in sorted(
            selected_by_pixel
        )
    ]

    return selected


def prepare_laser_lsfs(
    extracted_spectra: List[
        ExtractedLaserSpectrum
    ],
    ib_region_size: int,
    apply_325nm_repairs: bool = True,
    duplicate_source_policy: DuplicateSourcePolicy = "first",
) -> List[PreparedLaserLSF]:
    """
    Prepare the measured LSFs exactly as used by the original notebook.
    """

    selected_spectra = select_unique_source_spectra(
        extracted_spectra=extracted_spectra,
        duplicate_source_policy=duplicate_source_policy,
    )

    prepared_lsfs: List[
        PreparedLaserLSF
    ] = []

    for extracted in selected_spectra:
        corrected_spectrum = np.asarray(
            extracted.corrected_counts_per_cycle,
            dtype=np.float64,
        ).copy()

        if corrected_spectrum.size != 2048:
            raise ValueError(
                f"Expected 2048 detector pixels for "
                f"{extracted.wavelength_nm:g} nm."
            )

        if (
            apply_325nm_repairs
            and np.isclose(
                extracted.wavelength_nm,
                325.0,
            )
        ):
            corrected_spectrum = (
                apply_original_325nm_repairs(
                    corrected_spectrum
                )
            )

        normalized_lsf = normalize_lsf_min_max(
            corrected_spectrum
        )

        (
            normalized_sdf_column,
            ib_start,
            ib_end,
            ib_sum,
        ) = normalize_lsf_correct(
            normalized_lsf=normalized_lsf,
            pixel_number=int(
                extracted.peak_pixel_index
            ),
            ib_region_size=ib_region_size,
        )

        prepared_lsfs.append(
            PreparedLaserLSF(
                wavelength_nm=float(
                    extracted.wavelength_nm
                ),
                wavelength_occurrence=int(
                    extracted.wavelength_occurrence
                ),
                routine_count=int(
                    extracted.routine_count
                ),
                source_pixel=int(
                    extracted.peak_pixel_index
                ),
                corrected_spectrum=(
                    corrected_spectrum
                ),
                normalized_lsf=normalized_lsf,
                normalized_sdf_column=(
                    normalized_sdf_column
                ),
                ib_start=ib_start,
                ib_end_exclusive=ib_end,
                ib_sum=ib_sum,
            )
        )

    prepared_lsfs.sort(
        key=lambda item: item.source_pixel
    )

    return prepared_lsfs


def build_sdf_matrix_original_method(
    prepared_lsfs: List[
        PreparedLaserLSF
    ],
    detector_pixel_count: int = 2048,
    ib_region_size: int = 20,
) -> np.ndarray:
    """
    Construct the SDF matrix exactly as in Pandora63_analysis.ipynb.

    This method does not interpolate measured SDF values.

    Between measured source pixels, it shifts the measured column on
    the right upward as columns move left.
    """

    if not prepared_lsfs:
        raise ValueError(
            "No prepared laser LSFs were supplied."
        )

    source_pixels = [
        item.source_pixel
        for item in prepared_lsfs
    ]

    if source_pixels != sorted(source_pixels):
        raise ValueError(
            "Prepared LSFs must be sorted by source pixel."
        )

    if len(set(source_pixels)) != len(source_pixels):
        raise ValueError(
            "Prepared LSF source pixels must be unique."
        )

    sdf_matrix = np.zeros(
        (
            detector_pixel_count,
            detector_pixel_count,
        ),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Place normalized measured LSFs in their source columns
    # --------------------------------------------------------

    for prepared in prepared_lsfs:
        sdf_matrix[
            :,
            prepared.source_pixel,
        ] = prepared.normalized_sdf_column

    # --------------------------------------------------------
    # Shift LSFs upward moving left between measured columns
    # --------------------------------------------------------

    for index in range(
        len(source_pixels) - 1,
        0,
        -1,
    ):
        current_pixel = source_pixels[index]
        previous_pixel = source_pixels[index - 1]

        for column in range(
            current_pixel - 1,
            previous_pixel,
            -1,
        ):
            shift_amount = (
                current_pixel - column
            )

            sdf_matrix[
                :-shift_amount,
                column,
            ] = sdf_matrix[
                shift_amount:,
                current_pixel,
            ]

            sdf_matrix[
                -shift_amount:,
                column,
            ] = 0.0

    # --------------------------------------------------------
    # Shift the first measured LSF left to detector pixel zero
    # --------------------------------------------------------

    first_pixel = source_pixels[0]

    for column in range(
        first_pixel - 1,
        -1,
        -1,
    ):
        shift_amount = (
            first_pixel - column
        )

        sdf_matrix[
            :-shift_amount,
            column,
        ] = sdf_matrix[
            shift_amount:,
            first_pixel,
        ]

        sdf_matrix[
            -shift_amount:,
            column,
        ] = 0.0

    # --------------------------------------------------------
    # Shift the final measured LSF downward moving right
    # --------------------------------------------------------

    last_lsf_pixel = source_pixels[-1]

    for column in range(
        last_lsf_pixel + 1,
        detector_pixel_count,
    ):
        shift_amount = (
            column - last_lsf_pixel
        )

        sdf_matrix[
            shift_amount:,
            column,
        ] = sdf_matrix[
            :-shift_amount,
            last_lsf_pixel,
        ]

        sdf_matrix[
            :shift_amount,
            column,
        ] = 0.0

    # --------------------------------------------------------
    # Replace lower shifted-in zeros using the final row value
    # of each right-hand shifting LSF
    # --------------------------------------------------------

    for index in range(
        len(source_pixels) - 1,
        -1,
        -1,
    ):
        current_pixel = source_pixels[index]

        stop_column = (
            source_pixels[index - 1] + 1
            if index > 0
            else 0
        )

        last_value = sdf_matrix[
            -1,
            current_pixel,
        ]

        for column in range(
            current_pixel - 1,
            stop_column - 1,
            -1,
        ):
            ib_start = max(
                0,
                column - ib_region_size // 2,
            )

            ib_end = min(
                detector_pixel_count,
                column
                + ib_region_size // 2
                + 1,
            )

            # ib_start is intentionally retained here to match
            # the original code structure.
            _ = ib_start

            for row in range(
                ib_end,
                detector_pixel_count,
            ):
                if sdf_matrix[
                    row,
                    column,
                ] == 0:
                    sdf_matrix[
                        row,
                        column,
                    ] = last_value

    # --------------------------------------------------------
    # Replace upper shifted-in zeros to the right of the last
    # measured LSF using its first-row value
    # --------------------------------------------------------

    first_value = sdf_matrix[
        0,
        last_lsf_pixel,
    ]

    for column in range(
        last_lsf_pixel + 1,
        detector_pixel_count,
    ):
        ib_start = max(
            0,
            column - ib_region_size // 2,
        )

        ib_end = min(
            detector_pixel_count,
            column
            + ib_region_size // 2
            + 1,
        )

        # ib_end is intentionally retained to reproduce the
        # original code structure.
        _ = ib_end

        for row in range(
            0,
            ib_start,
        ):
            if sdf_matrix[
                row,
                column,
            ] == 0:
                sdf_matrix[
                    row,
                    column,
                ] = first_value

    return sdf_matrix


def build_original_notebook_matrices(
    extracted_spectra: List[
        ExtractedLaserSpectrum
    ],
    ib_region_size: int,
    detector_pixel_count: int = 2048,
    apply_325nm_repairs: bool = True,
    duplicate_source_policy: DuplicateSourcePolicy = "first",
) -> OriginalNotebookMatrixResult:
    """
    Execute the original Pandora63_analysis.ipynb matrix procedure.
    """

    prepared_lsfs = prepare_laser_lsfs(
        extracted_spectra=extracted_spectra,
        ib_region_size=ib_region_size,
        apply_325nm_repairs=apply_325nm_repairs,
        duplicate_source_policy=duplicate_source_policy,
    )

    sdf_matrix = build_sdf_matrix_original_method(
        prepared_lsfs=prepared_lsfs,
        detector_pixel_count=detector_pixel_count,
        ib_region_size=ib_region_size,
    )

    identity_matrix = np.eye(
        detector_pixel_count,
        dtype=np.float64,
    )

    a_matrix = (
        identity_matrix
        + sdf_matrix
    )

    correction_matrix = np.linalg.inv(
        a_matrix
    )

    source_pixels = np.asarray(
        [
            prepared.source_pixel
            for prepared in prepared_lsfs
        ],
        dtype=np.int64,
    )

    wavelengths_nm = np.asarray(
        [
            prepared.wavelength_nm
            for prepared in prepared_lsfs
        ],
        dtype=np.float64,
    )

    return OriginalNotebookMatrixResult(
        sdf_matrix=sdf_matrix,
        identity_matrix=identity_matrix,
        a_matrix=a_matrix,
        correction_matrix=correction_matrix,
        prepared_lsfs=prepared_lsfs,
        source_pixels=source_pixels,
        wavelengths_nm=wavelengths_nm,
        detector_pixel_count=detector_pixel_count,
        ib_region_size=ib_region_size,
    )