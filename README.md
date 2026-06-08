# Photonics Surrogate Modeling

Machine-learning and numerical surrogate models for predicting confinement-loss curves from refractive index values in a photonic sensor dataset.

The project uses simulated confinement-loss spectra where each sample is represented as:

```text
RI -> y_01, y_02, ..., y_61
```

The wavelength axis is fixed from `0.60` to `1.20` with 61 points. The model input is the refractive index, and the output is the full 61-point confinement-loss curve.

## Project Goal

The goal is to replace repeated simulation runs with a fast surrogate model that can predict the confinement-loss response for unseen refractive index values.

The key challenge is that the resonance peak does not only change height. It also shifts along the wavelength axis as refractive index changes. Direct curve prediction struggles with this peak movement, especially in the high-RI region.

## Dataset

Clean CSV files are stored in:

```text
dataset/training_clean.csv
dataset/testing_clean.csv
```

Expected format:

```text
ri,y_01,y_02,...,y_61
```

Current dataset:

```text
Training samples: 8
Testing samples: 4
Output points per curve: 61
```

## Best Model

The strongest machine-learning model in this project is:

```text
Peak-Aligned PCA-GPR
```

File:

```text
models/peak_aligned_pca_gpr.py
```

This is a hybrid machine-learning surrogate model:

1. Detect the resonance peak of each training curve.
2. Shift each curve so the peak is centered.
3. Normalize the peak height.
4. Use PCA to compress the aligned curve shape into 4 components.
5. Use Gaussian Process Regression to predict:
   - peak wavelength
   - peak confinement loss
   - PCA shape coefficients
6. Reconstruct the full predicted confinement-loss curve.
7. Compare predicted curves with actual testing curves.

## Current Best Result

After correcting testing samples 3 and 4, the selected model achieved:

```text
Model: Peak-Aligned PCA + Gaussian Process Regression
PCA components: 4
Explained variance: 97.956825%

Overall R2:   0.9822113145
Overall MAE:  0.003045989555
Overall RMSE: 0.007466306258
```

Per-sample R2:

```text
RI 1.3650  R2 = 0.98747524
RI 1.3833  R2 = 0.99153713
RI 1.3880  R2 = 0.96899022
RI 1.3921  R2 = 0.98519132
```

## Why Peak Alignment Matters

Direct models try to learn:

```text
RI -> 61 fixed y-values
```

This is difficult because the resonance peak moves along the x-axis. A fixed wavelength point may be on the left side of the peak for one RI value, at the peak for another, and on the right side for another.

Peak alignment separates the problem into easier parts:

```text
RI -> peak position
RI -> peak height
RI -> normalized curve shape
```

This makes the prediction more physically meaningful and improves the model's ability to reconstruct unseen curves.

## Scripts

### Main ML Models

```text
models/gpr.py
```

Direct Gaussian Process Regression model.

```text
models/pca_gpr.py
```

PCA-assisted GPR without peak alignment.

```text
models/peak_aligned_gpr.py
```

Peak alignment with direct GPR shape prediction, without PCA.

```text
models/peak_aligned_pca_gpr.py
```

Peak alignment with PCA shape compression and GPR prediction. This is the main selected ML model.

### Interpolation Models

```text
interpolation/spline_interpolation.py
```

Natural cubic spline interpolation model.

```text
interpolation/pchip_interpolation.py
```

PCHIP interpolation model.

```text
interpolation/peak_aligned_surrogate.py
```

Peak-aligned local interpolation surrogate.

```text
interpolation/width_aligned_peak_surrogate.py
```

Experimental width-aligned peak surrogate with data audit.

### Visualization

```text
visualization/plot_training_curves.py
visualization/plot_testing_curves.py
visualization/visualize_peak_alignment.py
```

These scripts show the raw curves and the peak-alignment transformation.

## Installation

This project uses `uv`.

```powershell
uv sync
```

Dependencies are defined in:

```text
pyproject.toml
```

## How To Use The App

This project is a script-based research app. You run one Python file at a time, and each script prints metrics and opens plots.

1. Install dependencies:

```powershell
uv sync
```

2. Run the selected best model:

```powershell
uv run models/peak_aligned_pca_gpr.py
```

3. Read the terminal output:

```text
R2, MAE, RMSE, per-RI accuracy, peak wavelength error, and peak loss error
```

4. View the plot window:

```text
solid line = actual testing curve
dashed line = predicted testing curve
```

The CSV files in `dataset/` are loaded automatically. No manual file selection is needed.

## What Is `photonics/core.py`?

`photonics/core.py` is the shared helper module for the project. It is not a separate model.

It contains common code used by the scripts:

```text
dataset loading
fixed wavelength axis
GPR fitting and prediction
PCA fitting and reconstruction
metric calculation
plotting helpers
```

Keeping this code in one place prevents every model file from repeating the same functions.

## Run The Best Model

```powershell
uv run models/peak_aligned_pca_gpr.py
```

The script prints:

```text
R2
MAE
RMSE
per-RI metrics
peak wavelength error
peak confinement-loss error
peak comparison table
```

It also displays actual-vs-predicted plots. The scripts do not save plot images by default.

## Run Other Models

```powershell
uv run models/gpr.py
uv run models/pca_gpr.py
uv run models/peak_aligned_gpr.py
uv run interpolation/spline_interpolation.py
uv run interpolation/pchip_interpolation.py
uv run interpolation/peak_aligned_surrogate.py
uv run interpolation/width_aligned_peak_surrogate.py
```

## Research Use

Recommended paper description:

```text
A peak-aligned PCA-GPR hybrid surrogate model was developed to predict confinement-loss spectra from refractive index values. The peak-alignment step compensates for resonance wavelength shifts, PCA reduces the dimensionality of the aligned curve shape, and GPR predicts the peak parameters and PCA coefficients for unseen refractive index values.
```

This model should be described as:

```text
physics-guided hybrid machine-learning surrogate model
```

not as a pure black-box ML model.

## Notes

- `R2` measures full-curve reconstruction quality.
- `MAE` and `RMSE` measure absolute prediction error.
- Peak wavelength error and peak loss error are important for sensor-specific evaluation.
- More simulation data near fast peak-shift regions can further improve accuracy.
