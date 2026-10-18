"""Part 2a -- one peak-normalised KDE map per class.

Saves surfaces.npz, which Parts 3 and 4 read.  The stored maps use the
16x16 lookup grid while the figure uses a fine display grid.
"""

import matplotlib.pyplot as plt
import numpy as np

from pipeline_common import (
    VIEW_LIM,
    evaluate_on_grid,
    fit_density,
    load_split_scaled,
    make_grid,
    normalise_peak,
)


BANDWIDTH = 0.05
DISPLAY_N = 200

X = load_split_scaled()["train"]
labels = np.loadtxt("labels.csv", dtype=int)
classes = sorted(set(labels) - {-1})
xx, yy, grid_points = make_grid()

kdes = {c: fit_density(X[labels == c], BANDWIDTH) for c in classes}
surfaces = {
    c: np.rint(
        normalise_peak(evaluate_on_grid(kdes[c], grid_points, xx)) * 255
    ).astype(np.uint8)
    for c in classes
}
np.savez("surfaces.npz", **{f"c{c}": surface for c, surface in surfaces.items()})

g = np.linspace(0, 1, DISPLAY_N)
fxx, fyy = np.meshgrid(g, g)
fine = np.column_stack([fxx.ravel(), fyy.ravel()])

fig, axes = plt.subplots(
    1, len(classes), figsize=(3.0 * len(classes), 3.0),
    sharex=True, sharey=True,
)
for ax, c in zip(np.atleast_1d(axes), classes):
    smooth = normalise_peak(
        np.exp(kdes[c].score_samples(fine)).reshape(fxx.shape)
    )
    im = ax.imshow(
        smooth, extent=[0, 1, 0, 1], origin="lower", cmap="Greys",
        interpolation="bilinear", aspect="equal", vmin=0, vmax=1,
    )
    ax.scatter(
        X[labels == c][:, 0], X[labels == c][:, 1], s=2,
        c="white", edgecolors="black", linewidths=0.2, alpha=0.7,
    )
    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)
    ax.set_title(f"class {c}, continuous", fontsize=9)
    ax.set_xlabel("scaled feature $x_1$")
np.atleast_1d(axes)[0].set_ylabel("scaled feature $x_2$")
fig.colorbar(im, ax=axes, shrink=0.85, label="normalised density")
fig.savefig(
    "../figures/generated/class_surfaces_continuous.pdf",
    bbox_inches="tight", transparent=True,
)
