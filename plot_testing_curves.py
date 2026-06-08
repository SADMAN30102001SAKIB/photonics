from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "dataset" / "testing_clean.csv"


def load_testing_data(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    y_columns = [f"y_{index:02d}" for index in range(1, 62)]
    return [
        (row["ri"], [float(row[column]) for column in y_columns])
        for row in rows
    ]


def main():
    x_values = [0.6 + index * 0.01 for index in range(61)]
    curves = load_testing_data(DATA_FILE)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    for ri, y_values in curves:
        ax.plot(x_values, y_values, linewidth=2, label=f"RI = {ri}")

    ax.set_title("Testing Dataset Confinement Loss Curves", fontsize=16, weight="bold")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Confinement Loss")
    ax.set_xlim(0.6, 1.2)
    ax.legend(title="Refractive Index", ncols=2, frameon=True)

    plt.show()


if __name__ == "__main__":
    main()
