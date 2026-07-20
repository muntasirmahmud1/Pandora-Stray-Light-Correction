# Pandora-Stray-Light-Correction


The objective of this project is to estimate and remove optical stray
light in Pandora spectrometer measurements using monochromatic laser
calibration data. The measured laser line spread functions (LSFs) are
converted into stray-light distribution functions (SDFs), assembled into
a stray-light matrix, and inverted to generate a correction matrix that
is applied to operational Level-0 (L0) spectra.

## Mathematical Model

Let:

-   $\mathbf{x}$ = true detector signal
-   $\mathbf{y}$ = measured detector signal
-   $\mathbf{S}$ = stray-light matrix

The measured spectrum is

$$
\mathbf{y}=\mathbf{x}+\mathbf{S}\mathbf{x}
$$

or

$$
\mathbf{y}=(\mathbf{I}+\mathbf{S})\mathbf{x}.
$$

Therefore,

$$
\mathbf{x}=(\mathbf{I}+\mathbf{S})^{-1}\mathbf{y},
$$

where

$$
\mathbf{C}=(\mathbf{I}+\mathbf{S})^{-1}
$$

is the correction matrix.

## Calibration Procedure

### Step 1 -- Laser Calibration

Acquire monochromatic laser measurements spanning the detector
wavelength range.

### Step 2 -- Automatic Laser Extraction

Automatically identify laser exposures, pair bright and dark
measurements, and compute

$$
L=L_{\rm bright}-L_{\rm dark}.
$$

### Step 3 -- Normalize the LSF

$$
L_n(i)=
\frac{L(i)-L_{\min}}
{L_{\max}-L_{\min}}.
$$

### Step 4 -- Define the In-band Region

For peak pixel $i_{peak}$ and half-width $k$,

$$
i_{peak}-k\le i\le i_{peak}+k.
$$

### Step 5 -- Compute the Stray-light Distribution

The in-band pixels are excluded and the remaining signal is normalized:

$$
SDF(i)=
\frac{L_n(i)}
{\sum_{IB}L_n},
\qquad i\notin IB.
$$

### Step 6 -- Build the Stray-light Matrix

Each measured SDF is inserted into the detector column corresponding to
its laser peak. Intermediate columns are generated using the same
shifting algorithm as the original Pandora notebook.

### Step 7 -- Compute the Correction Matrix

$$
\mathbf{A}=\mathbf{I}+\mathbf{S}
$$

$$
\mathbf{C}=\mathbf{A}^{-1}
$$

## L0 File Correction before trace gas retrieval

Dark counts are first removed,

$$
\mathbf{y_d}=\mathbf{y}-\mathbf{d},
$$

the correction matrix is applied,

$$
\mathbf{x_d}=\mathbf{C}\mathbf{y_d},
$$

negative values are clipped to zero,

$$
\mathbf{x_d}=\max(\mathbf{x_d},0),
$$

and the dark counts are restored,

$$
\boxed{\mathbf{x}_{corr}=\mathbf{d}+\max\left(\mathbf{C}(\mathbf{y}-\mathbf{d}),0\right)}
$$

## Validation

The corrected L0 spectra are processed through the standard Pandora
L1/L2 retrieval chain and compared against a reference Pandora
instrument. Recommended validation plots include:

-   Raw vs corrected spectra
-   Normalized LSFs
-   Stray-light matrix
-   Correction matrix
-   Ozone retrieval comparison
-   Before/after retrieval residuals

## Workflow

``` text
Calibration L0
    ↓
Automatic laser detection
    ↓
Bright/Dark pairing
    ↓
Laser LSF extraction
    ↓
Normalized SDF
    ↓
Stray-light Matrix (S)
    ↓
Correction Matrix C=(I+S)^−1
    ↓
Operational L0
    ↓
Corrected L0
    ↓
L1/L2 Retrieval
    ↓
Validation
```
