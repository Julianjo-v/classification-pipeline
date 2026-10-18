"""Create the required true-label by DBSCAN-cluster confusion table."""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import load_split_scaled


HERE = Path(__file__).resolve().parent


def main():
    split = load_split_scaled()
    true_labels = split["y_train"]
    cluster_labels = np.loadtxt(HERE / "labels.csv", dtype=int)
    if len(true_labels) != len(cluster_labels):
        raise ValueError("labels.csv length does not match the training split")

    table = pd.crosstab(
        pd.Series(true_labels, name="true_label"),
        pd.Series(cluster_labels, name="dbscan_cluster"),
        dropna=False,
    ).reindex(sorted(set(cluster_labels)), axis=1, fill_value=0)
    output = HERE / "dbscan_ground_truth_confusion.csv"
    table.to_csv(output)
    print("DBSCAN versus ground truth (rows=true label, columns=cluster):")
    print(table.to_string())
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
