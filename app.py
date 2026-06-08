from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Tool:
    name: str
    path: Path
    description: str


TOOLS = [
    Tool(
        "Peak-Aligned PCA-GPR",
        Path("models/peak_aligned_pca_gpr.py"),
        "Best selected ML model with peak alignment, PCA, and GPR.",
    ),
    Tool(
        "GPR",
        Path("models/gpr.py"),
        "Direct Gaussian Process Regression on the full curve.",
    ),
    Tool(
        "PCA-GPR",
        Path("models/pca_gpr.py"),
        "PCA curve compression with GPR coefficient prediction.",
    ),
    Tool(
        "Peak-Aligned GPR",
        Path("models/peak_aligned_gpr.py"),
        "Peak alignment with direct GPR shape prediction.",
    ),
    Tool(
        "PCHIP Interpolation",
        Path("interpolation/pchip_interpolation.py"),
        "Shape-preserving interpolation comparison method.",
    ),
    Tool(
        "Natural Cubic Spline",
        Path("interpolation/spline_interpolation.py"),
        "Cubic spline interpolation comparison method.",
    ),
    Tool(
        "Peak-Aligned Interpolation",
        Path("interpolation/peak_aligned_surrogate.py"),
        "Peak-aligned normalized-shape interpolation.",
    ),
    Tool(
        "Width-Aligned Interpolation",
        Path("interpolation/width_aligned_peak_surrogate.py"),
        "Peak and width aligned interpolation experiment.",
    ),
    Tool(
        "Plot Training Curves",
        Path("visualization/plot_training_curves.py"),
        "Plot all cleaned training curves.",
    ),
    Tool(
        "Plot Testing Curves",
        Path("visualization/plot_testing_curves.py"),
        "Plot all cleaned testing curves.",
    ),
    Tool(
        "Visualize Peak Alignment",
        Path("visualization/visualize_peak_alignment.py"),
        "Show how peak alignment transforms a curve.",
    ),
]


def print_menu() -> None:
    print("\nPhotonics CLI")
    print("=" * 40)
    for index, tool in enumerate(TOOLS, start=1):
        print(f"{index:2}. {tool.name}")
        print(f"    {tool.description}")
    print(" q. Quit")


def read_choice() -> int | None:
    while True:
        value = input("\nSelect a tool: ").strip().lower()

        if value in {"q", "quit", "exit"}:
            return None

        if not value.isdigit():
            print("Please enter a number from the menu, or q to quit.")
            continue

        choice = int(value)
        if 1 <= choice <= len(TOOLS):
            return choice - 1

        print(f"Please enter a number from 1 to {len(TOOLS)}, or q to quit.")


def run_tool(tool: Tool) -> int:
    script = ROOT / tool.path
    if not script.exists():
        print(f"Missing script: {tool.path}")
        return 1

    print(f"\nRunning: {tool.name}")
    print(f"Script:  {tool.path}\n")

    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    return result.returncode


def main() -> int:
    while True:
        print_menu()
        choice = read_choice()

        if choice is None:
            print("Goodbye.")
            return 0

        exit_code = run_tool(TOOLS[choice])
        if exit_code != 0:
            print(f"\nTool exited with code {exit_code}.")

        again = input("\nRun another tool? [y/N]: ").strip().lower()
        if again != "y":
            return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
