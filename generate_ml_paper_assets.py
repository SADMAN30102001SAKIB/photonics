from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from models.peak_aligned_gpr import fit_peak_aligned_gpr, predict_peak_aligned_gpr
from photonics.core import TESTING_FILE, TRAINING_FILE, X_AXIS, load_dataset


OUT_DIR = Path("Research_Soumik")


def smooth_curve(x, y, points=400):
    x_smooth = np.linspace(x.min(), x.max(), points)
    y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
    return x_smooth, np.maximum(y_smooth, 0.0)


def peak_info(curve):
    index = np.argmax(curve)
    return index, X_AXIS[index], curve[index]


def save_peak_alignment():
    ri, y = load_dataset(TRAINING_FILE, expected_rows=8)
    selected = [5, 6, 7]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    for index in selected:
        curve = y[index]
        _, peak_wavelength, peak_loss = peak_info(curve)
        shifted_x = X_AXIS - peak_wavelength

        x_original, y_original = smooth_curve(X_AXIS, curve)
        x_shifted, y_shifted = smooth_curve(shifted_x, curve)
        _, y_normalized = smooth_curve(shifted_x, curve / peak_loss)

        axes[0].plot(x_original, y_original, linewidth=2, label=f"RI = {ri[index]:.2f}")
        axes[0].scatter(peak_wavelength, peak_loss, s=42)

        axes[1].plot(x_shifted, y_shifted, linewidth=2, label=f"RI = {ri[index]:.2f}")
        axes[1].scatter(0, peak_loss, s=42)

        axes[2].plot(
            x_shifted,
            y_normalized,
            linewidth=2,
            label=f"RI = {ri[index]:.2f}",
        )
        axes[2].scatter(0, 1, s=42)

    axes[0].set_title("(a) Original spectra")
    axes[0].set_xlabel("Wavelength (um)")
    axes[0].set_ylabel("Confinement loss")

    axes[1].set_title("(b) Peak shifted to zero")
    axes[1].set_xlabel("Wavelength - peak wavelength (um)")
    axes[1].set_ylabel("Confinement loss")
    axes[1].axvline(0, color="black", linewidth=1, alpha=0.5)

    axes[2].set_title("(c) Peak height normalized")
    axes[2].set_xlabel("Wavelength - peak wavelength (um)")
    axes[2].set_ylabel("Normalized confinement loss")
    axes[2].axvline(0, color="black", linewidth=1, alpha=0.5)
    axes[2].axhline(1, color="black", linewidth=1, alpha=0.5)

    for ax in axes:
        ax.legend(fontsize=8)

    fig.savefig(OUT_DIR / "ML_peak_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_peak_aligned_prediction():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    model = fit_peak_aligned_gpr(train_ri, train_y)
    predicted_y = predict_peak_aligned_gpr(model, test_ri)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True)

    for ax, ri, actual_curve, predicted_curve in zip(
        axes.ravel(),
        test_ri,
        test_y,
        predicted_y,
    ):
        actual_x, actual_smooth = smooth_curve(X_AXIS, actual_curve)
        predicted_x, predicted_smooth = smooth_curve(X_AXIS, predicted_curve)

        ax.plot(actual_x, actual_smooth, linewidth=2.1, label="Actual")
        ax.plot(
            predicted_x,
            predicted_smooth,
            linewidth=2.1,
            linestyle="--",
            label="Predicted",
        )
        ax.set_title(f"RI = {ri:.4f}", fontsize=11, weight="bold")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Confinement loss")
        ax.set_xlim(0.6, 1.2)
        ax.legend(fontsize=8)

    fig.savefig(
        OUT_DIR / "ML_peak_aligned_gpr_prediction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    save_peak_alignment()
    save_peak_aligned_prediction()


if __name__ == "__main__":
    main()
