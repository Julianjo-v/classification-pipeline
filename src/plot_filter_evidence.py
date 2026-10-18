"""Plot stored sensor readings against the firmware EMA filter.

The combined CSV is treated as the raw input sequence. If the readings were
already filtered during collection, describe them that way in the report.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data_complete.csv"
DEFAULT_OUTPUT = HERE.parent / "figures" / "generated" / "raw_vs_filtered_sun.pdf"


def ema(values, alpha):
    filtered = np.empty_like(values, dtype=float)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--class-name", default="sun")
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not 0.0 < args.alpha <= 1.0:
        parser.error("--alpha must be greater than 0 and at most 1")

    data = pd.read_csv(args.data)
    required = {"uv_sensor", "IR_sensor", "label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    selected = data[data["label"].astype(str).str.casefold() == args.class_name.casefold()]
    if selected.empty:
        available = sorted(data["label"].astype(str).unique())
        raise ValueError(f"class {args.class_name!r} not found; available: {available}")

    sample = np.arange(len(selected))
    signals = [
        ("UV sensor", selected["uv_sensor"].to_numpy(dtype=float)),
        ("IR sensor", selected["IR_sensor"].to_numpy(dtype=float)),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.7), sharex=True)
    for axis, (name, raw) in zip(axes, signals):
        filtered = ema(raw, args.alpha)
        raw_step = np.max(np.abs(np.diff(raw)))
        filtered_step = np.max(np.abs(np.diff(filtered)))

        axis.plot(sample, raw, color="0.68", linewidth=0.8, label="Stored reading (raw input)")
        axis.plot(sample, filtered, color="black", linewidth=1.3,
                  label=f"EMA filtered (alpha={args.alpha:.2f})")
        axis.set_ylabel("Sensor reading")
        axis.set_title(name, loc="left", fontsize=10)
        axis.grid(True, color="0.9", linewidth=0.5)
        axis.text(
            0.99, 0.05,
            f"maximum step: raw {raw_step:.2f}, filtered {filtered_step:.2f}",
            transform=axis.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        axis.legend(frameon=False, loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(
        f"Raw input versus EMA-filtered signal - {args.class_name} class (n={len(selected)})",
        fontsize=11,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        bbox_inches="tight",
        metadata={
            "Title": "Raw input versus EMA-filtered sensor signals",
            "Subject": f"Filtering evidence for the {args.class_name} class",
        },
    )
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
