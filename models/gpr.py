from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonics.core import (
    TESTING_FILE,
    TRAINING_FILE,
    fit_sklearn_gpr,
    load_dataset,
    plot_predictions,
    predict_sklearn_gpr,
    print_standard_results,
)


def main():
    train_ri, train_y = load_dataset(TRAINING_FILE, expected_rows=8)
    test_ri, test_y = load_dataset(TESTING_FILE, expected_rows=4)

    model = fit_sklearn_gpr(train_ri, train_y)
    predicted_y = predict_sklearn_gpr(model, test_ri)

    print_standard_results(
        "Scikit-learn Gaussian Process Regression surrogate model",
        test_ri,
        test_y,
        predicted_y,
        extra_lines=[
            f"Kernel:           {model['model'].kernel_}",
            f"Log likelihood:   {model['model'].log_marginal_likelihood_value_:.8f}",
        ],
    )
    plot_predictions("Actual vs Predicted Testing Curves", test_ri, test_y, predicted_y)


if __name__ == "__main__":
    main()
