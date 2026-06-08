import matplotlib.pyplot as plt
import numpy as np

from main import (
    TESTING_FILE,
    TRAINING_FILE,
    X_AXIS,
    fit_gpr,
    load_dataset,
    peak_metrics,
    predict_gpr,
    regression_metrics,
)

ALIGNED_X = np.linspace(-0.35, 0.35, 141)


def extract_peaks(y):
    peak_index = np.argmax(y, axis=1)
    peak_wavelength = X_AXIS[peak_index]
    peak_loss = y[np.arange(len(y)), peak_index]
    return peak_wavelength, peak_loss


def align_curve(y_curve, peak_wavelength, peak_loss):
    shifted_x = X_AXIS - peak_wavelength
    normalized_y = y_curve / peak_loss
    return np.interp(ALIGNED_X, shifted_x, normalized_y, left=0.0, right=0.0)


def fit_peak_aligned_gpr(train_ri, train_y):
    peak_wavelength, peak_loss = extract_peaks(train_y)
    aligned_shapes = np.array(
        [
            align_curve(curve, wavelength, loss)
            for curve, wavelength, loss in zip(train_y, peak_wavelength, peak_loss)
        ]
    )

    peak_model = fit_gpr(
        train_ri,
        np.column_stack([peak_wavelength, np.log(peak_loss)]),
    )
    shape_model = fit_gpr(train_ri, aligned_shapes)

    return {
        "peak_model": peak_model,
        "shape_model": shape_model,
    }


def predict_peak_aligned_gpr(model, test_ri):
    predicted_peak = predict_gpr(model["peak_model"], test_ri)
    predicted_wavelength = predicted_peak[:, 0]
    predicted_loss = np.exp(predicted_peak[:, 1])
    aligned_shapes = predict_gpr(model["shape_model"], test_ri)

    predictions = []
    for wavelength, loss, aligned_shape in zip(
        predicted_wavelength,
        predicted_loss,
        aligned_shapes,
    ):
        shifted_x = X_AXIS - wavelength
        normalized_curve = np.interp(
            shifted_x,
            ALIGNED_X,
            aligned_shape,
            left=0.0,
            right=0.0,
        )
        predictions.append(np.maximum(normalized_curve * loss, 0.0))

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

    print("Peak-Aligned Gaussian Process Regression")
    print("=" * 58)
    print("Training samples: 8")
    print(f"Testing samples:  {len(test_ri)}")
    print("Output points:    61")
    print("Strategy: peak alignment + direct GPR shape prediction")
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

    fig.suptitle("Peak-Aligned GPR: Actual vs Predicted", fontsize=16, weight="bold")
    plt.show()


def main():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    model = fit_peak_aligned_gpr(train_ri, train_y)
    predicted_y = predict_peak_aligned_gpr(model, test_ri)

    print_results(test_ri, test_y, predicted_y)
    plot_predictions(test_ri, test_y, predicted_y)


if __name__ == "__main__":
    main()
