"""
analyze_stability_and_plot.py

Run the analysis and plotting pipeline for the paper.

The script:
  1. Builds combined probing prediction tables.
  2. Builds combined zero-shot prediction tables.
  3. Builds retraction/expansion summary tables.
  4. Creates Figure 2, Figure 3/SI, and LLM-level SI plots.
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT / "analyze_plot_scripts"


def run_step(command: list[str]) -> None:
    """Run one pipeline command from the project root."""
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(command))
    print("=" * 100)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def script_path(name: str) -> str:
    """Return path to an analysis/plotting script."""
    path = SCRIPT_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Could not find required script: {path}")

    return str(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subprocesses.",
    )
    parser.add_argument(
        "--datasets",
        default="cities_loc,med_indications,defs",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip table-building scripts and only run plotting scripts.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Run table-building scripts but skip plotting scripts.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the full analysis and plotting pipeline."""
    args = parse_args()
    py = args.python

    if not args.skip_build:
        for probe in ["sAwMIL", "mean_diff"]:
            run_step(
                [
                    py,
                    script_path("build_results_summary_probes.py"),
                    "--probe",
                    probe,
                    "--datasets",
                    args.datasets,
                ]
            )

        run_step(
            [
                py,
                script_path("build_results_summary_zeroshot.py"),
                "--datasets",
                args.datasets,
            ]
        )

        for probe in ["sAwMIL", "mean_diff"]:
            run_step(
                [
                    py,
                    script_path("build_retraction_expansion_tables_probes.py"),
                    "--probe",
                    probe,
                    "--datasets",
                    args.datasets,
                ]
            )

        run_step(
            [
                py,
                script_path("build_retraction_expansion_tables_zeroshot.py"),
                "--datasets",
                args.datasets,
            ]
        )

    if not args.skip_plots:
        run_step([py, script_path("plot_fig2.py")])
        run_step([py, script_path("plot_fig3+SI.py")])
        run_step([py, script_path("plot_SI_llm_level.py")])


if __name__ == "__main__":
    main()