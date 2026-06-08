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


COMPONENTS = 6


def fit_pca(y, components):
    mean = y.mean(axis=0)
    centered = y - mean
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    basis = vectors[:components]
    coefficients = centered @ basis.T
    variance = singular_values**2 / (len(y) - 1)
    explained = variance[:components] / variance.sum()
    return mean, basis, coefficients, explained


def reconstruct_pca(mean, basis, coefficients):
    return coefficients @ basis + mean


def print_results(test_ri, actual, predicted, explained):
    overall_r2, overall_mae, overall_rmse = regression_metrics(actual, predicted)
    peak_wavelength_error, peak_loss_error, actual_wl, predicted_wl, actual_loss, predicted_loss = peak_metrics(
        actual,
        predicted,
    )

    print("PCA + Gaussian Process Regression")
    print("=" * 52)
    print("Training samples: 8")
    print(f"Testing samples:  {len(test_ri)}")
    print("Output points:    61")
    print(f"PCA components:   {len(explained)}")
    print(f"Explained variance: {explained.sum() * 100:.6f}%")
    print()
    print("PCA explained variance by component")
    print("-" * 52)
    for index, value in enumerate(explained, start=1):
        print(f"PC{index}: {value * 100:.6f}%")

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

    fig.suptitle("PCA + GPR: Actual vs Predicted Testing Curves", fontsize=16, weight="bold")
    plt.show()


def main():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    pca_mean, pca_basis, train_coefficients, explained = fit_pca(train_y, COMPONENTS)
    gpr_model = fit_gpr(train_ri, train_coefficients)
    predicted_coefficients = predict_gpr(gpr_model, test_ri)
    predicted_y = reconstruct_pca(pca_mean, pca_basis, predicted_coefficients)

    print_results(test_ri, test_y, predicted_y, explained)
    plot_predictions(test_ri, test_y, predicted_y)


if __name__ == "__main__":
    main()
