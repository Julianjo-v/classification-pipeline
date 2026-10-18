
# SECTION 1: IMPORT LIBRARIES

import matplotlib.pyplot as plt
import numpy as np
from pipeline_common import VIEW_LIM, load_split_scaled
from sklearn.cluster import DBSCAN, KMeans


# SECTION 2: LOAD THE SENSOR DATASET

# Load, split, and scale the UV and IR sensor dataset
data = load_split_scaled()

# Only use training data for clustering
X = data["train"]


# SECTION 3: FIND THE BEST DBSCAN PARAMETERS

# Search for eps and min_samples values that produce exactly 4 clusters
# eps controls the maximum distance between neighboring points
# min_samples controls how many nearby points are needed to form a dense area

# Start with no selected parameters or labels
best_eps, best_min_samples, best_labels = None, None, None

# Test 100 different eps values between 0.01 and 0.15
for eps in np.linspace(0.01, 0.15, 100):

  # Test min_samples values from 3 to 14
  for min_samples in range(3, 15):

    # Create the DBSCAN model using the current parameters
    db = DBSCAN(eps=eps, min_samples=min_samples)

    # Apply DBSCAN to the training data and get cluster labels
    labels = db.fit_predict(X)

    # Count the clusters while excluding -1 (noise points)
    classes = sorted(set(labels) - {-1})

    # Check if DBSCAN found exactly 4 clusters
    if len(classes) == 4:

      # Save the parameters and labels that produced 4 clusters
      best_eps = eps
      best_min_samples = min_samples
      best_labels = labels
      break

  # Stop searching once a valid result is found
  if best_labels is not None:
    break


# SECTION 4: APPLY THE SELECTED DBSCAN RESULT

# Check if the automatic search found exactly 4 clusters
if best_labels is not None:

  # Display the selected parameters
  print(f"Found 4 clusters with eps={best_eps:.3f}, min_samples={best_min_samples}")

  # Use the labels from the successful DBSCAN result
  labels = best_labels

else:

  # If no exact match is found, use the default parameters
  print("Exact 4 clusters not found automatically; using tuned default (eps=0.04, min_samples=5).")

  # Create and run DBSCAN using the default settings
  db = DBSCAN(eps=0.04, min_samples=5)
  labels = db.fit_predict(X)


# SECTION 5: DISPLAY CLUSTER INFORMATION

# Get the cluster numbers while excluding noise (-1)
classes = sorted(set(labels) - {-1})

# Display the total number of clusters and noise points
print(f"{len(classes)} clusters; {(labels == -1).sum()} noise")

# Display how many points belong to each cluster
for c in classes:
  print(f" cluster {c}: {(labels == c).sum()} points")


# SECTION 6: SAVE THE DBSCAN LABELS

# Save the cluster labels to a CSV file
# Other scripts can use these labels later
np.savetxt("labels.csv", labels, fmt="%d")


# SECTION 7: APPLY K-MEANS FOR COMPARISON

# Create a K-Means model using the number of clusters found by DBSCAN
# If DBSCAN found no clusters, use 1 cluster instead
# n_init=10 means K-Means tries 10 different starting positions
# random_state=0 makes the results reproducible

kmeans_labels = KMeans(
    n_clusters=len(classes) if len(classes) > 0 else 1, n_init=10, random_state=0
).fit_predict(X)


# SECTION 8: PLOT SETTINGS

# Different marker shapes used to identify each cluster
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Different shades of gray used for each cluster
GRAYS = ["0.75", "0.45", "0.15", "0.6", "0.3", "0.85"]


# SECTION 9: FUNCTION TO DRAW THE CLUSTERS

def draw(ax, lab, title, show_noise):

  # Go through each cluster while excluding noise (-1)
  for i, c in enumerate(sorted(set(lab) - {-1})):

    # Select all points that belong to the current cluster
    pts = X[lab == c]

    # Plot the cluster points using different markers and gray shades
    ax.scatter(
        pts[:, 0],  # X coordinates (first sensor feature)
        pts[:, 1],  # Y coordinates (second sensor feature)
        s=5,
        marker=MARKERS[i % len(MARKERS)],
        facecolor=GRAYS[i % len(GRAYS)],
        edgecolor="black",
        linewidth=0.25,
        label=f"class {c} (n={len(pts)})",
    )

  # Check if noise points should also be displayed
  if show_noise:

    # Select points labeled -1 by DBSCAN
    noise = X[lab == -1]

    # Plot noise points using an X marker
    ax.scatter(
        noise[:, 0],
        noise[:, 1],
        s=9,
        c="0.45",
        marker="x",
        linewidth=0.5,
        label=f"noise, $-1$ (n={len(noise)})",
    )

  # Set the graph title and X-axis label
  ax.set_title(title, fontsize=9)
  ax.set_xlabel("scaled feature $x_1$")

  # Keep both axes at the same scale
  ax.set_aspect("equal")

  # Set the visible range of both axes
  ax.set_xlim(*VIEW_LIM)
  ax.set_ylim(*VIEW_LIM)



# SECTION 10: COMPARE DBSCAN AND K-MEANS=

# Create two graphs side by side
# Both graphs share the same X and Y axis scales
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True, sharey=True)

# Left graph: DBSCAN keeps irregular cluster shapes and shows noise
draw(axes[0], labels, "DBSCAN: shapes stay intact", show_noise=True)

# Right graph: K-Means divides the data into clusters based on centroids
# Unlike DBSCAN, K-Means assigns every point to a cluster
draw(axes[1], kmeans_labels, "k-means: shapes get sliced", show_noise=False)

# Set the Y-axis label on the left graph
axes[0].set_ylabel("scaled feature $x_2$")


# SECTION 11: SAVE THE COMPARISON GRAPH

# Save both graphs together as a PDF file
# bbox_inches removes unnecessary whitespace
# transparent=True saves the graph with a transparent background
fig.savefig(
    "../figures/generated/clustered_space.pdf",
    bbox_inches="tight",
    transparent=True,)
