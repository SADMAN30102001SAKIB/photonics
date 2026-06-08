from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonics.core import (
    TRAINING_FILE,
    TESTING_FILE,
    X_AXIS,
    load_dataset,
    peak_metrics,
    regression_metrics,
)


def fit_natural_cubic_spline(x, y):
    n = len(x)
    intervals = np.diff(x)
    slopes = np.diff(y, axis=0) / intervals[:, None]

    system = np.zeros((n, n))
    rhs = np.zeros((n, y.shape[1]))
    system[0, 0] = 1.0
    system[-1, -1] = 1.0

    for row in range(1, n - 1):
        system[row, row - 1] = intervals[row - 1]
        system[row, row] = 2 * (intervals[row - 1] + intervals[row])
        system[row, row + 1] = intervals[row]
        rhs[row] = 6 * (slopes[row] - slopes[row - 1])

    second_derivatives = np.linalg.solve(system, rhs)
    return x, y, second_derivatives


def predict_natural_cubic_spline(model, x_test):
    x_train, y_train, second_derivatives = model
    predictions = []

    for x_value in x_test:
        index = np.searchsorted(x_train, x_value) - 1
        index = np.clip(index, 0, len(x_train) - 2)

        left = x_train[index]
        right = x_train[index + 1]
        width = right - left
        a = (right - x_value) / width
        b = (x_value - left) / width

        y_value = (
            a * y_train[index]
            + b * y_train[index + 1]
            + (
                (a**3 - a) * second_derivatives[index]
                + (b**3 - b) * second_derivatives[index + 1]
            )
            * (width**2)
            / 6
        )
        predictions.append(y_value)

    return np.array(predictions)


def print_results(test_ri, actual, predicted):
    overall_r2, overall_mae, overall_rmse = regression_metrics(actual, predicted)
    (
        peak_wavelength_error,
        peak_loss_error,
        actual_wl,
        predicted_wl,
        actual_loss,
        predicted_loss,
    ) = peak_metrics(
        actual,
        predicted,
    )

    print("Natural Cubic Spline interpolation")
    print("=" * 52)
    print("Training samples: 8")
    print(f"Testing samples:  {len(test_ri)}")
    print("Output points:    61")
    print()
    print("Overall testing accuracy")
    print("-" * 52)
    print(f"R2:   {overall_r2:.10f}")
    print(f"MAE:  {overall_mae:.12f}")
    print(f"RMSE: {overall_rmse:.12f}")
    print()
    print("Per-RI testing accuracy")
    print("-" * 112)
    print(
        "RI       R2            MAE             RMSE            "
        "Peak WL Error   Peak Loss Error"
    )
    print("-" * 112)

    for index, ri in enumerate(test_ri):
        r2, mae, rmse = regression_metrics(actual[index], predicted[index])
        print(
            f"{ri:<8.4f} "
            f"{r2:>12.8f} "
            f"{mae:>15.12f} "
            f"{rmse:>15.12f} "
            f"{peak_wavelength_error[index]:>15.4f} "
            f"{peak_loss_error[index]:>17.12f}"
        )

    print()
    print("Peak comparison")
    print("-" * 86)
    print("RI       Actual WL   Predicted WL   Actual Peak Loss   Predicted Peak Loss")
    print("-" * 86)
    for index, ri in enumerate(test_ri):
        print(
            f"{ri:<8.4f} "
            f"{actual_wl[index]:>9.2f} "
            f"{predicted_wl[index]:>14.2f} "
            f"{actual_loss[index]:>18.12f} "
            f"{predicted_loss[index]:>21.12f}"
        )


def plot_predictions(test_ri, actual, predicted):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for ax, ri, actual_curve, predicted_curve in zip(
        axes.ravel(),
        test_ri,
        actual,
        predicted,
    ):
        ax.plot(X_AXIS, actual_curve, linewidth=2.2, label="Actual")
        ax.plot(
            X_AXIS, predicted_curve, linewidth=2.2, linestyle="--", label="Predicted"
        )
        ax.set_title(f"RI = {ri:.4f}", fontsize=12, weight="bold")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Confinement Loss")
        ax.set_xlim(0.6, 1.2)
        ax.legend()

    fig.suptitle(
        "Cubic Spline: Actual vs Predicted Testing Curves", fontsize=16, weight="bold"
    )
    plt.show()


def main():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    model = fit_natural_cubic_spline(train_ri, train_y)
    predicted_y = predict_natural_cubic_spline(model, test_ri)

    print_results(test_ri, test_y, predicted_y)
    plot_predictions(test_ri, test_y, predicted_y)


if __name__ == "__main__":
    main()
