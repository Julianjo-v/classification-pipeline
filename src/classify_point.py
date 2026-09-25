"""Part 4 -- classify one held-out query from stored byte tables only."""

import matplotlib.pyplot as plt
import numpy as np

from pipeline_common import (
    GRID_N,
    REJECT_TAU,
    VIEW_LIM,
    cell_of,
    dequantize,
    load_split_scaled,
    quantize,
    scale_features,
)


data = load_split_scaled()
Xtest, bounds = data["test"], data["bounds"]
surfaces = np.load("surfaces.npz")
classes = sorted(int(key[1:]) for key in surfaces.files)
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}


def _win_value(point):
    return max(dequantize(tables[c][cell_of(point)]) for c in classes)


_interior = [
    i for i, point in enumerate(Xtest)
    if _win_value(point) >= REJECT_TAU
    and all(2 <= value <= GRID_N - 3 for value in cell_of(point))
]
QUERY = Xtest[max(_interior, key=lambda i: _win_value(Xtest[i]))]
QUERY_RAW = QUERY * (bounds[1] - bounds[0]) + bounds[0]


def lookup_nearest(point):
    """One byte per class from the cell containing the point."""
    row, col = cell_of(point)
    return np.array([dequantize(tables[c][row, col]) for c in classes])


def lookup_bilinear(point):
    """Blend four neighbouring cells: four bytes per class."""
    col_float, row_float = np.clip(np.asarray(point) * GRID_N - 0.5, 0, GRID_N - 1)
    col0, row0 = int(col_float), int(row_float)
    col1, row1 = min(col0 + 1, GRID_N - 1), min(row0 + 1, GRID_N - 1)
    col_fraction, row_fraction = col_float - col0, row_float - row0
    out = []
    for c in classes:
        table = dequantize(tables[c])
        out.append(
            table[row0, col0] * (1 - col_fraction) * (1 - row_fraction)
            + table[row0, col1] * col_fraction * (1 - row_fraction)
            + table[row1, col0] * (1 - col_fraction) * row_fraction
            + table[row1, col1] * col_fraction * row_fraction
        )
    return np.array(out)


def classify(values):
    """Distribution over classes, or None when the winner is below tau."""
    total = values.sum()
    distribution = values / total if total > 0 else np.zeros_like(values)
    best = int(np.argmax(values))
    return (None if values[best] < REJECT_TAU else classes[best]), distribution


for name, values in (("nearest", lookup_nearest(QUERY)), ("bilinear", lookup_bilinear(QUERY))):
    label, distribution = classify(values)
    verdict = "unknown" if label is None else f"class {label}"
    print(f"{name:>9}: {verdict:>9} bytes=" + " ".join(f"{value * 255:6.1f}" for value in values)
          + " dist=" + " ".join(f"{value * 100:5.1f}%" for value in distribution))

row, col = cell_of(QUERY)
print(f"\nquery {QUERY_RAW} -> cell [row {row}, col {col}] of {GRID_N}x{GRID_N}")
print("stored bytes in the 3x3 window around that cell:")
for c in classes:
    print(f" class {c}:")
    for r in range(min(row + 1, GRID_N - 1), max(row - 2, -1), -1):
        print(" " + " ".join(f"{tables[c][r, cc]:4d}" for cc in range(max(col - 1, 0), min(col + 2, GRID_N))))

stack = np.stack([dequantize(tables[c]) for c in classes])
winning_value = stack.max(axis=0)
rejected = winning_value < REJECT_TAU
print(f"\nreject region: {rejected.sum()} of {rejected.size} cells ({100 * rejected.mean():.1f}%) below tau={REJECT_TAU}")

ring_centre, _ = scale_features(np.array([4.8, 1.6]), bounds)
label, _ = classify(lookup_nearest(ring_centre))
print(f"ring centre [4.8 1.6] -> {'unknown' if label is None else f'class {label}'}")

_, distribution = classify(lookup_nearest(QUERY))
winner = int(np.argmax(distribution))
edges = np.linspace(0, 1, GRID_N + 1)
fig, (grid_ax, probabilities_ax) = plt.subplots(1, 2, figsize=(7.2, 3.4))
grid_ax.add_patch(plt.Rectangle((edges[col], edges[row]), 1 / GRID_N, 1 / GRID_N,
                                facecolor="0.8", edgecolor="black", lw=1.2, zorder=1))
for edge in edges:
    grid_ax.axvline(edge, color="0.75", lw=0.2, zorder=0)
    grid_ax.axhline(edge, color="0.75", lw=0.2, zorder=0)
grid_ax.plot(*QUERY, marker="*", markersize=18, markerfacecolor="white",
             markeredgecolor="black", markeredgewidth=1.5, linestyle="none", zorder=3)
grid_ax.set(xlim=VIEW_LIM, ylim=VIEW_LIM, aspect="equal", xlabel="scaled feature $x_1$",
            ylabel="scaled feature $x_2$", title=f"Test query in cell (row {row}, col {col})")
positions = np.arange(len(classes))
shades = ["0.8"] * len(classes)
shades[winner] = "0.2"
probabilities_ax.bar(positions, distribution, width=0.6, color=shades, edgecolor="black", linewidth=0.8)
for position, probability in zip(positions, distribution):
    probabilities_ax.text(position, probability + 0.03, f"{probability:.2f}", ha="center", va="bottom", fontsize=9)
probabilities_ax.axhline(REJECT_TAU, color="black", ls="--", lw=0.8)
probabilities_ax.text(len(classes) - 0.5, REJECT_TAU + 0.01, f"$\\tau={REJECT_TAU}$", ha="right", va="bottom", fontsize=8)
probabilities_ax.set(xticks=positions, xticklabels=[f"class {c}" for c in classes], ylim=(0, 1.15),
                     ylabel="probability $p_c$", title=f"argmax $\\rightarrow$ class {classes[winner]}")
for side in ("top", "right"):
    probabilities_ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("../figures/generated/decision_map.pdf", bbox_inches="tight", transparent=True)
