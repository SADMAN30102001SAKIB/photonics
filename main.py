from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAINING_FILE = ROOT / "dataset" / "training_clean.csv"
TESTING_FILE = ROOT / "dataset" / "testing_clean.csv"
X_AXIS = np.linspace(0.6, 1.2, 61)


def load_dataset(path, expected_rows):
    y_columns = [f"y_{index:02d}" for index in range(1, 62)]

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != expected_rows:
        raise ValueError(f"{path.name} has {len(rows)} rows; expected {expected_rows}.")

    expected_columns = ["ri", *y_columns]
    if list(rows[0].keys()) != expected_columns:
        raise ValueError(
            f"{path.name} does not match the expected ri,y_01,...,y_61 schema."
        )

    ri = np.array([float(row["ri"]) for row in rows], dtype=float)
    y = np.array(
        [[float(row[column]) for column in y_columns] for row in rows],
        dtype=float,
    )

    if not np.isfinite(ri).all() or not np.isfinite(y).all():
        raise ValueError(f"{path.name} contains missing or non-numeric values.")

    return ri, y


def rbf_kernel(x_a, x_b, length_scale):
    distances = (x_a[:, None] - x_b[None, :]) ** 2
    return np.exp(-0.5 * distances / (length_scale**2))


def log_marginal_likelihood(x_train, y_train, length_scale, noise):
    kernel = rbf_kernel(x_train, x_train, length_scale)
    kernel += noise * np.eye(len(x_train))

    try:
        lower = np.linalg.cholesky(kernel)
    except np.linalg.LinAlgError:
        return -np.inf

    weights = np.linalg.solve(lower.T, np.linalg.solve(lower, y_train))
    data_fit = -0.5 * np.sum(y_train * weights)
    complexity = -y_train.shape[1] * np.sum(np.log(np.diag(lower)))
    normalizer = -0.5 * len(x_train) * y_train.shape[1] * np.log(2 * np.pi)
    return data_fit + complexity + normalizer


def fit_gpr(x_train, y_train):
    x_mean = x_train.mean()
    x_std = x_train.std()
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0

    x_scaled = (x_train - x_mean) / x_std
    y_scaled = (y_train - y_mean) / y_std

    best_score = -np.inf
    best_length_scale = None
    best_noise = None

    for length_scale in np.logspace(-1.5, 1.0, 80):
        for noise in np.logspace(-10, -3, 8):
            score = log_marginal_likelihood(x_scaled, y_scaled, length_scale, noise)
            if score > best_score:
                best_score = score
                best_length_scale = length_scale
                best_noise = noise

    kernel = rbf_kernel(x_scaled, x_scaled, best_length_scale)
    kernel += best_noise * np.eye(len(x_scaled))
    lower = np.linalg.cholesky(kernel)
    weights = np.linalg.solve(lower.T, np.linalg.solve(lower, y_scaled))

    return {
        "x_train": x_scaled,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "length_scale": best_length_scale,
        "noise": best_noise,
        "weights": weights,
    }


def predict_gpr(model, x_test):
    x_scaled = (x_test - model["x_mean"]) / model["x_std"]
    cross_kernel = rbf_kernel(x_scaled, model["x_train"], model["length_scale"])
    y_scaled = cross_kernel @ model["weights"]
    return y_scaled * model["y_std"] + model["y_mean"]


def regression_metrics(actual, predicted):
    errors = predicted - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2, mae, rmse


def peak_metrics(actual, predicted):
    actual_peak_index = np.argmax(actual, axis=1)
    predicted_peak_index = np.argmax(predicted, axis=1)

    actual_peak_wavelength = X_AXIS[actual_peak_index]
    predicted_peak_wavelength = X_AXIS[predicted_peak_index]
    actual_peak_loss = actual[np.arange(len(actual)), actual_peak_index]
    predicted_peak_loss = predicted[np.arange(len(predicted)), predicted_peak_index]

    return (
        np.abs(predicted_peak_wavelength - actual_peak_wavelength),
        np.abs(predicted_peak_loss - actual_peak_loss),
        actual_peak_wavelength,
        predicted_peak_wavelength,
        actual_peak_loss,
        predicted_peak_loss,
    )


def print_results(test_ri, actual, predicted, model):
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

    print("Gaussian Process Regression surrogate model")
    print("=" * 52)
    print(f"Training samples: 8")
    print(f"Testing samples:  {len(test_ri)}")
    print(f"Output points:    61")
    print(f"Length scale:     {model['length_scale']:.8f}")
    print(f"Noise:            {model['noise']:.2e}")
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

    fig.suptitle("Actual vs Predicted Testing Curves", fontsize=16, weight="bold")
    plt.show()


def main():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    model = fit_gpr(train_ri, train_y)
    predicted_y = predict_gpr(model, test_ri)

    print_results(test_ri, test_y, predicted_y, model)
    plot_predictions(test_ri, test_y, predicted_y)


if __name__ == "__main__":
    main()
