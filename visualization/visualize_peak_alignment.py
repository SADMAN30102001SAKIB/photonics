from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonics.core import TRAINING_FILE, X_AXIS, load_dataset


def peak_info(curve):
    index = np.argmax(curve)
    return index, X_AXIS[index], curve[index]


def main():
    ri, y = load_dataset(TRAINING_FILE, expected_rows=8)
    selected = [5, 6, 7]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    for index in selected:
        curve = y[index]
        peak_index, peak_wavelength, peak_loss = peak_info(curve)
        axes[0].plot(X_AXIS, curve, linewidth=2, label=f"RI = {ri[index]:.2f}")
        axes[0].scatter(peak_wavelength, peak_loss, s=45)
        axes[0].annotate(
            f"peak",
            (peak_wavelength, peak_loss),
            textcoords="offset points",
            xytext=(5, 6),
            fontsize=9,
        )

        shifted_x = X_AXIS - peak_wavelength
        axes[1].plot(shifted_x, curve, linewidth=2, label=f"RI = {ri[index]:.2f}")
        axes[1].scatter(0, peak_loss, s=45)

        normalized_y = curve / peak_loss
        axes[2].plot(shifted_x, normalized_y, linewidth=2, label=f"RI = {ri[index]:.2f}")
        axes[2].scatter(0, 1, s=45)

    axes[0].set_title("1. Original Curves")
    axes[0].set_xlabel("Wavelength")
    axes[0].set_ylabel("Confinement Loss")

    axes[1].set_title("2. Shift Peak to Zero")
    axes[1].set_xlabel("Wavelength - Peak Wavelength")
    axes[1].set_ylabel("Confinement Loss")
    axes[1].axvline(0, color="black", linewidth=1, alpha=0.5)

    axes[2].set_title("3. Normalize Peak Height")
    axes[2].set_xlabel("Wavelength - Peak Wavelength")
    axes[2].set_ylabel("Confinement Loss / Peak Loss")
    axes[2].axvline(0, color="black", linewidth=1, alpha=0.5)
    axes[2].axhline(1, color="black", linewidth=1, alpha=0.5)

    for ax in axes:
        ax.legend()

    fig.suptitle("What Peak Alignment Does", fontsize=16, weight="bold")
    plt.show()


if __name__ == "__main__":
    main()
