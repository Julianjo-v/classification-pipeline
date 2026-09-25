"""Part 4 -- spend the test set once on the frozen design.

Reports agreement between the stored uint8 lookup and the continuous KDE on
unseen points; this is not a ground-truth accuracy score because DBSCAN
invented the labels.
"""

import numpy as np

from pipeline_common import (
    REJECT_TAU,
    cell_of,
    dequantize,
    fit_density,
    load_split_scaled,
    make_grid,
    normalise_peak,
    quantize,
)


BANDWIDTH = 0.05
data = load_split_scaled()
Xtrain, Xtest = data["train"], data["test"]
labels = np.loadtxt("labels.csv", dtype=int)
classes = sorted(set(labels) - {-1})
xx, yy, grid_points = make_grid()

kdes, peaks = {}, {}
for c in classes:
    kde = fit_density(Xtrain[labels == c], BANDWIDTH)
    kdes[c] = kde
    peaks[c] = np.exp(kde.score_samples(grid_points)).max()

surfaces = np.load("surfaces.npz")
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}


def continuous_decision(point):
    values = np.array([
        np.exp(kdes[c].score_samples([point])[0]) / peaks[c]
        for c in classes
    ])
    best = int(np.argmax(values))
    return -1 if values[best] < REJECT_TAU else classes[best]


def stored_decision(point):
    row, col = cell_of(point)
    values = np.array([dequantize(tables[c][row, col]) for c in classes])
    best = int(np.argmax(values))
    return -1 if values[best] < REJECT_TAU else classes[best]


continuous = np.array([continuous_decision(point) for point in Xtest])
stored = np.array([stored_decision(point) for point in Xtest])
agreement = (continuous == stored).mean()
print(f"test points: {len(Xtest)}")
print(f"stored-vs-continuous agreement on held-out test: {100 * agreement:.1f}%")
print("\nconfusion (rows = continuous KDE, cols = stored uint8):")
all_labels = classes + [-1]
header = "".join(f"{('unk' if label == -1 else 'c' + str(label)):>6}" for label in all_labels)
print(f"{'':>10}{header}")
for left in all_labels:
    row = "".join(f"{int(((continuous == left) & (stored == right)).sum()):>6}" for right in all_labels)
    name = "unknown" if left == -1 else f"class {left}"
    print(f"{name:>10}{row}")
