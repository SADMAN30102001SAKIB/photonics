from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonics.core import (
    TESTING_FILE,
    TRAINING_FILE,
    X_AXIS,
    load_dataset,
    peak_metrics,
    regression_metrics,
)

ALIGNED_X = np.linspace(-5.0, 5.0, 301)


def extract_peak_features(y):
    peak_index = np.argmax(y, axis=1)
    peak_wavelength = X_AXIS[peak_index]
    peak_loss = y[np.arange(len(y)), peak_index]

    left_width = []
    right_width = []

    for curve, index, peak in zip(y, peak_index, peak_loss):
        half_peak = peak / 2
        left_candidates = np.where(curve[: index + 1] <= half_peak)[0]
        right_candidates = np.where(curve[index:] <= half_peak)[0] + index

        left_index = left_candidates[-1] if len(left_candidates) else 0
        right_index = right_candidates[0] if len(right_candidates) else len(curve) - 1

        left_width.append(max(X_AXIS[index] - X_AXIS[left_index], 0.01))
        right_width.append(max(X_AXIS[right_index] - X_AXIS[index], 0.01))

    return (
        peak_index,
        peak_wavelength,
        peak_loss,
        np.array(left_width),
        np.array(right_width),
    )


def aligned_coordinate(x_values, peak_wavelength, left_width, right_width):
    shifted = x_values - peak_wavelength
    width = np.where(shifted < 0, left_width, right_width)
    return shifted / width


def aligned_shape(curve, peak_wavelength, peak_loss, left_width, right_width):
    coordinate = aligned_coordinate(X_AXIS, peak_wavelength, left_width, right_width)
    normalized_curve = curve / peak_loss
    return np.interp(ALIGNED_X, coordinate, normalized_curve, left=0.0, right=0.0)


def interpolate_log(left, right, fraction):
    return np.exp((1 - fraction) * np.log(left) + fraction * np.log(right))


def predict_curves(train_ri, train_y, test_ri):
    _, peak_wavelength, peak_loss, left_width, right_width = extract_peak_features(
        train_y
    )
    aligned_shapes = np.array(
        [
            aligned_shape(curve, wavelength, loss, left, right)
            for curve, wavelength, loss, left, right in zip(
                train_y,
                peak_wavelength,
                peak_loss,
                left_width,
                right_width,
            )
        ]
    )

    predictions = []

    for ri in test_ri:
        left_index = np.searchsorted(train_ri, ri) - 1
        left_index = np.clip(left_index, 0, len(train_ri) - 2)
        right_index = left_index + 1
        fraction = (ri - train_ri[left_index]) / (
            train_ri[right_index] - train_ri[left_index]
        )

        predicted_peak_wavelength = (1 - fraction) * peak_wavelength[
            left_index
        ] + fraction * peak_wavelength[right_index]
        predicted_peak_loss = interpolate_log(
            peak_loss[left_index],
            peak_loss[right_index],
            fraction,
        )
        predicted_left_width = interpolate_log(
            left_width[left_index],
            left_width[right_index],
            fraction,
        )
        predicted_right_width = interpolate_log(
            right_width[left_index],
            right_width[right_index],
            fraction,
        )
        predicted_shape = (1 - fraction) * aligned_shapes[
            left_index
        ] + fraction * aligned_shapes[right_index]

        target_coordinate = aligned_coordinate(
            X_AXIS,
            predicted_peak_wavelength,
            predicted_left_width,
            predicted_right_width,
        )
        normalized_curve = np.interp(
            target_coordinate,
            ALIGNED_X,
            predicted_shape,
            left=0.0,
            right=0.0,
        )
        predictions.append(np.maximum(normalized_curve * predicted_peak_loss, 0.0))

    return np.array(predictions)


def print_data_audit(train_ri, train_y, test_ri, test_y):
    _, train_wl, train_loss, train_left, train_right = extract_peak_features(train_y)
    _, test_wl, test_loss, test_left, test_right = extract_peak_features(test_y)

    print("Data audit")
    print("=" * 52)
    print("Training peak trend")
    print("-" * 52)
    print("RI       Peak WL   Peak Loss       Left HW   Right HW")
    for ri, wl, loss, left, right in zip(
        train_ri, train_wl, train_loss, train_left, train_right
    ):
        print(f"{ri:<8.4f} {wl:>7.2f}   {loss:>13.12f}   {left:>7.2f}   {right:>8.2f}")

    print()
    print("Testing peak summary")
    print("-" * 52)
    print("RI       Peak WL   Peak Loss       Left HW   Right HW")
    for ri, wl, loss, left, right in zip(
        test_ri, test_wl, test_loss, test_left, test_right
    ):
        print(f"{ri:<8.4f} {wl:>7.2f}   {loss:>13.12f}   {left:>7.2f}   {right:>8.2f}")

    print()
    print("Audit notes")
    print("-" * 52)
    print(f"Training RI range: {train_ri.min():.4f} to {train_ri.max():.4f}")
    print(f"Testing RI range:  {test_ri.min():.4f} to {test_ri.max():.4f}")
    print(
        f"All testing RI values inside training range: {bool(np.all((test_ri >= train_ri.min()) & (test_ri <= train_ri.max())))}"
    )
    print(
        f"Training peak wavelength monotonic increasing: {bool(np.all(np.diff(train_wl) >= 0))}"
    )
    print(
        f"Training peak loss monotonic increasing:       {bool(np.all(np.diff(train_loss) >= 0))}"
    )
    print()


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

    print("Width-aligned peak surrogate model")
    print("=" * 52)
    print("Training samples: 8")
    print(f"Testing samples:  {len(test_ri)}")
    print("Output points:    61")
    print(
        "Strategy: local peak alignment + log peak scaling + asymmetric width alignment"
    )
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
        "Width-Aligned Peak Surrogate: Actual vs Predicted", fontsize=16, weight="bold"
    )
    plt.show()


def main():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    print_data_audit(train_ri, train_y, test_ri, test_y)
    predicted_y = predict_curves(train_ri, train_y, test_ri)
    print_results(test_ri, test_y, predicted_y)
    plot_predictions(test_ri, test_y, predicted_y)


if __name__ == "__main__":
    main()
