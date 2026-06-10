from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from models.peak_aligned_gpr import fit_peak_aligned_gpr, predict_peak_aligned_gpr
from photonics.core import TESTING_FILE, TRAINING_FILE, X_AXIS, load_dataset


OUT_DIR = Path("Research_Soumik")
TITLE_SIZE = 24
AXIS_LABEL_SIZE = 26
TICK_LABEL_SIZE = 23
LEGEND_SIZE = 23
LINE_WIDTH = 3.6
MARKER_SIZE = 110


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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)

    for index in selected:
        curve = y[index]
        _, peak_wavelength, peak_loss = peak_info(curve)
        shifted_x = X_AXIS - peak_wavelength

        x_original, y_original = smooth_curve(X_AXIS, curve)
        x_shifted, y_shifted = smooth_curve(shifted_x, curve)
        _, y_normalized = smooth_curve(shifted_x, curve / peak_loss)

        axes[0].plot(
            x_original,
            y_original,
            linewidth=LINE_WIDTH,
            label=f"RI = {ri[index]:.2f}",
        )
        axes[0].scatter(peak_wavelength, peak_loss, s=MARKER_SIZE)

        axes[1].plot(
            x_shifted,
            y_shifted,
            linewidth=LINE_WIDTH,
            label=f"RI = {ri[index]:.2f}",
        )
        axes[1].scatter(0, peak_loss, s=MARKER_SIZE)

        axes[2].plot(
            x_shifted,
            y_normalized,
            linewidth=LINE_WIDTH,
            label=f"RI = {ri[index]:.2f}",
        )
        axes[2].scatter(0, 1, s=MARKER_SIZE)

    axes[0].set_title("(a) Original", fontsize=TITLE_SIZE, weight="bold")
    axes[0].set_xlabel("Wavelength (um)", fontsize=AXIS_LABEL_SIZE, weight="bold")
    axes[0].set_ylabel("CL (dB/cm)", fontsize=AXIS_LABEL_SIZE, weight="bold")

    axes[1].set_title("(b) Shifted peak", fontsize=TITLE_SIZE, weight="bold")
    axes[1].set_xlabel("Centered wavelength (um)", fontsize=AXIS_LABEL_SIZE, weight="bold")
    axes[1].set_ylabel("CL (dB/cm)", fontsize=AXIS_LABEL_SIZE, weight="bold")
    axes[1].axvline(0, color="black", linewidth=1.8, alpha=0.5)

    axes[2].set_title("(c) Normalized", fontsize=TITLE_SIZE, weight="bold")
    axes[2].set_xlabel("Centered wavelength (um)", fontsize=AXIS_LABEL_SIZE, weight="bold")
    axes[2].set_ylabel("Normalized CL", fontsize=AXIS_LABEL_SIZE, weight="bold")
    axes[2].axvline(0, color="black", linewidth=1.8, alpha=0.5)
    axes[2].axhline(1, color="black", linewidth=1.8, alpha=0.5)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        legend = ax.legend(fontsize=LEGEND_SIZE, frameon=True)
        for text in legend.get_texts():
            text.set_fontweight("bold")

    fig.savefig(
        OUT_DIR / "ML_peak_alignment.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


def save_peak_aligned_prediction():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    model = fit_peak_aligned_gpr(train_ri, train_y)
    predicted_y = predict_peak_aligned_gpr(model, test_ri)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.8), constrained_layout=True)

    for ax, ri, actual_curve, predicted_curve in zip(
        axes.ravel(),
        test_ri,
        test_y,
        predicted_y,
    ):
        actual_x, actual_smooth = smooth_curve(X_AXIS, actual_curve)
        predicted_x, predicted_smooth = smooth_curve(X_AXIS, predicted_curve)

        ax.plot(actual_x, actual_smooth, linewidth=LINE_WIDTH, label="Actual")
        ax.plot(
            predicted_x,
            predicted_smooth,
            linewidth=LINE_WIDTH,
            linestyle="--",
            label="Predicted",
        )
        ax.set_title(f"RI = {ri:.4f}", fontsize=TITLE_SIZE, weight="bold")
        ax.set_xlabel("Wavelength (um)", fontsize=AXIS_LABEL_SIZE, weight="bold")
        ax.set_ylabel("CL (dB/cm)", fontsize=AXIS_LABEL_SIZE, weight="bold")
        ax.set_xlim(0.6, 1.2)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        legend = ax.legend(fontsize=LEGEND_SIZE, frameon=True)
        for text in legend.get_texts():
            text.set_fontweight("bold")

    fig.savefig(
        OUT_DIR / "ML_peak_aligned_gpr_prediction.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


def main():
    save_peak_alignment()
    save_peak_aligned_prediction()


if __name__ == "__main__":
    main()
