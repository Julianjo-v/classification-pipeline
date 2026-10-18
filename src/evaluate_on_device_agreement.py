"""Compare a fresh ESP32 serial log with the desktop byte-table lookup.

Save Serial Monitor output to a text file, then run:
    python evaluate_on_device_agreement.py device_serial_log.txt

The script reads the existing single-line summaries containing X1, X2, Class,
and Score. It does not modify the firmware.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
LINE_PATTERN = re.compile(
    r"X1:\s*(?P<x1>[-+0-9.eE]+)\s*\|\s*"
    r"X2:\s*(?P<x2>[-+0-9.eE]+)\s*\|\s*"
    r"Class:\s*(?P<device_class>[^|]+?)\s*\|\s*"
    r"Score:\s*(?P<device_score>\d+)"
)
CLASS_NAMES = ["LED", "FLUORESCENT", "SHADE", "SUN"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "serial_log",
        type=Path,
        nargs="?",
        default=Path(r"C:\Users\Admin\Documents\SCHOOL\classification-pipeline1\src\serial_log.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "on_device_vs_desktop_agreement.csv",
    )
    args = parser.parse_args()

    matches = [
        match.groupdict()
        for line in args.serial_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if (match := LINE_PATTERN.search(line))
    ]
    if not matches:
        raise ValueError("no ESP32 classification summary lines were found")

    with np.load(HERE / "surfaces.npz") as archive:
        tables = [archive[key].astype(np.uint8) for key in sorted(archive.files)]

    rows = []
    for item in matches:
        x1 = float(item["x1"])
        x2 = float(item["x2"])
        col = int(np.clip(x1 * tables[0].shape[1], 0, tables[0].shape[1] - 1))
        row = int(np.clip(x2 * tables[0].shape[0], 0, tables[0].shape[0] - 1))
        scores = [int(table[row, col]) for table in tables]
        desktop_class = CLASS_NAMES[int(np.argmax(scores))]
        device_class = item["device_class"].strip().upper()
        rows.append(
            {
                "x1": x1,
                "x2": x2,
                "device_class": device_class,
                "desktop_class": desktop_class,
                "device_score": int(item["device_score"]),
                "desktop_score": max(scores),
                "agree": device_class == desktop_class,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    agreement = frame["agree"].mean()
    print(f"fresh held-out queries: {len(frame)}")
    print(f"on-device versus desktop agreement: {agreement:.2%}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()