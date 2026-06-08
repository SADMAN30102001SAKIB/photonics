import matplotlib.pyplot as plt
import numpy as np

from main import (
    TESTING_FILE,
    TRAINING_FILE,
    X_AXIS,
    load_dataset,
    peak_metrics,
    regression_metrics,
)


def fit_pchip(x, y):
    h = np.diff(x)
    delta = np.diff(y, axis=0) / h[:, None]
    derivatives = np.zeros_like(y)

    for index in range(1, len(x) - 1):
        previous_delta = delta[index - 1]
        next_delta = delta[index]
        same_sign = previous_delta * next_delta > 0
        w1 = 2 * h[index] + h[index - 1]
        w2 = h[index] + 2 * h[index - 1]
        derivatives[index, same_sign] = (w1 + w2) / (
            (w1 / previous_delta[same_sign]) + (w2 / next_delta[same_sign])
        )

    derivatives[0] = endpoint_derivative(h[0], h[1], delta[0], delta[1])
    derivatives[-1] = endpoint_derivative(h[-1], h[-2], delta[-1], delta[-2])
    return x, y, derivatives


def endpoint_derivative(h_current, h_next, delta_current, delta_next):
    derivative = ((2 * h_current + h_next) * delta_current - h_current * delta_next) / (
        h_current + h_next
    )
    derivative[derivative * delta_current <= 0] = 0

    too_large = (delta_current * delta_next < 0) & (
        np.abs(derivative) > 3 * np.abs(delta_current)
    )
    derivative[too_large] = 3 * delta_current[too_large]
    return derivative


def predict_pchip(model, x_test):
    x_train, y_train, derivatives = model
    predictions = []

    for x_value in x_test:
        index = np.searchsorted(x_train, x_value) - 1
        index = np.clip(index, 0, len(x_train) - 2)

        left = x_train[index]
        right = x_train[index + 1]
        width = right - left
        t = (x_value - left) / width

        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        y_value = (
            h00 * y_train[index]
            + h10 * width * derivatives[index]
            + h01 * y_train[index + 1]
            + h11 * width * derivatives[index + 1]
        )
        predictions.append(y_value)

    return np.array(predictions)


def print_results(test_ri, actual, predicted):
    overall_r2, overall_mae, overall_rmse = regression_metrics(actual, predicted)
    peak_wavelength_error, peak_loss_error, actual_wl, predicted_wl, actual_loss, predicted_loss = peak_metrics(
        actual,
        predicted,
    )

    print("PCHIP interpolation")
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
        ax.plot(X_AXIS, predicted_curve, linewidth=2.2, linestyle="--", label="Predicted")
        ax.set_title(f"RI = {ri:.4f}", fontsize=12, weight="bold")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Confinement Loss")
        ax.set_xlim(0.6, 1.2)
        ax.legend()

    fig.suptitle("PCHIP: Actual vs Predicted Testing Curves", fontsize=16, weight="bold")
    plt.show()


def main():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    model = fit_pchip(train_ri, train_y)
    predicted_y = predict_pchip(model, test_ri)

    print_results(test_ri, test_y, predicted_y)
    plot_predictions(test_ri, test_y, predicted_y)


if __name__ == "__main__":
    main()
