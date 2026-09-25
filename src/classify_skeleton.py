"""Part 4 skeleton -- table-lookup classification from stored bytes."""

import numpy as np

from pipeline_common import (
    REJECT_TAU,
    GRID_N,
    dequantize,
    load_split_scaled,
    quantize,
    scale_features,
)


bounds = load_split_scaled()["bounds"]
surfaces = np.load("surfaces.npz")
classes = sorted(int(key[1:]) for key in surfaces.files)
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}


def classify(raw_point):
    """Class label for a raw point, or None for unknown."""
    point, _ = scale_features(np.asarray(raw_point), bounds)
    col, row = np.clip((point * GRID_N).astype(int), 0, GRID_N - 1)
    values = np.array([dequantize(tables[c][row, col]) for c in classes])
    best = int(np.argmax(values))
    return None if values[best] < REJECT_TAU else classes[best]


if __name__ == "__main__":
    for query in ([0.0, -1.8], [0.0, 5.0], [6.8, 1.6], [4.8, 1.6]):
        print(query, "->", classify(query))
