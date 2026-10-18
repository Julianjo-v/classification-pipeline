
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import pandas as pd

# SECTION 1: GLOBAL SETTINGS AND PARAMETERS

# Fixed random seed so the results stay the same each time
SEED = 20250903

# Grid size used to divide the feature space into cells
# Flash memory cost = number of classes * GRID_N**2 bytes
GRID_N = 16

# Minimum confidence needed to accept a classification
# If the value is below 0.12, the result is "unknown"
REJECT_TAU = 0.12

# Axis limits used when plotting the data
# Extra space is added around the [0,1] range
VIEW_LIM = (-0.1, 1.1)

# Divide the dataset into training, validation, and testing
# Training = 60%, Validation = 20%, Testing = 20%
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20


# SECTION 2: GENERATING POINTS ON AN ARC

def _arc(rng, cx, cy, r, a0_deg, a1_deg, n, jitter):
    """Generate noisy points along an arc or circle."""

    # Generate evenly spaced angles and convert degrees to radians
    a = np.linspace(np.radians(a0_deg), np.radians(a1_deg), n)

    # Calculate the X and Y coordinates of each point
    pts = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])

    # Add random noise to make the points less perfect
    return pts + rng.normal(0.0, jitter, pts.shape)


# SECTION 3: LOADING THE SENSOR DATASET

def load_dataset(csv_path=r"C:\Users\Admin\Documents\SCHOOL\classification-pipeline1\src\data_complete.csv"):
    """Load the UV and IR sensor readings and their labels."""

    # Read the CSV file containing the sensor data
    df = pd.read_csv(csv_path)

    # Extract UV and IR readings as input features
    X = df[["uv_sensor", "IR_sensor"]].to_numpy().astype(float)

    # Apply logarithmic scaling to the IR readings
    # This reduces the effect of very large IR values
    X[:, 1] = np.log1p(X[:, 1])

    # Extract the actual labels for each reading
    y = df["label"].to_numpy()

    # Return sensor readings and labels separately
    return X, y


# SECTION 4: ASSIGNING NAMES TO DBSCAN CLUSTERS

def cluster_names(labels, y):
    """Assign each cluster the most common true label inside it."""

    # Dictionary used to store each cluster's name
    names = {}

    # Go through each cluster except -1, which represents noise
    for c in sorted(set(labels) - {-1}):

        # Count how many times each true label appears in the cluster
        vals, counts = np.unique(y[labels == c], return_counts=True)

        # Use the most common label as the cluster name
        names[c] = str(vals[counts.argmax()])

    return names

# SECTION 5: FEATURE SCALING

def scale_features(X, bounds=None):
    """Scale each feature between 0 and 1 using min-max scaling."""

    # Find minimum and maximum values if bounds are not provided
    # Otherwise, reuse the bounds from the training dataset
    lo, hi = bounds if bounds is not None else (X.min(axis=0), X.max(axis=0))

    # Apply min-max scaling to the features
    return (X - lo) / (hi - lo), (lo, hi)


# SECTION 6: SPLITTING THE DATASET

def split_data(X, y, seed=SEED):
    """Split the dataset into 60% training, 20% validation, and 20% testing."""

    # First split: 60% for training and 40% for the remaining data
    # Stratify keeps the class proportions similar in each split
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=TRAIN_FRAC, random_state=seed, stratify=y)

    # Second split: divide the remaining 40% equally
    # 20% validation and 20% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)

    # Return all three datasets with their labels
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# SECTION 7: LOADING, SPLITTING, AND SCALING THE DATA

def load_split_scaled(seed=SEED):
    """Load the dataset, split it, and scale using training bounds."""

    # Load all UV and IR sensor readings
    X, y = load_dataset()

    # Divide the dataset into training, validation, and testing
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, seed)

    # Scale the training data and save its minimum and maximum values
    train_s, bounds = scale_features(X_train)

    # Scale validation and testing using the SAME training bounds
    # This prevents information from the other datasets leaking into training
    val_s, _ = scale_features(X_val, bounds)
    test_s, _ = scale_features(X_test, bounds)

    # Return all the scaled datasets and labels
    return {
        "train": train_s, "y_train": y_train,
        "val": val_s, "y_val": y_val,
        "test": test_s, "y_test": y_test,
        "bounds": bounds,
    }


# SECTION 8: CREATING THE GRID

def make_grid():
    """Create a GRID_N x GRID_N grid covering the [0,1] feature space."""

    # Calculate the center position of each grid cell
    centres = (np.arange(GRID_N) + 0.5) / GRID_N

    # Create X and Y coordinates for all grid cells
    xx, yy = np.meshgrid(centres, centres)

    # Return the grid coordinates and a list of grid points
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


# SECTION 9: FINDING WHICH GRID CELL CONTAINS A POINT

def cell_of(point):
    """Find the row and column of a scaled point."""

    # Convert the point coordinates into grid indices
    # Clip prevents indices from going outside the grid
    i = np.clip((np.asarray(point) * GRID_N).astype(int), 0, GRID_N - 1)

    # Return row and column
    return int(i[1]), int(i[0])


# SECTION 10: KERNEL DENSITY ESTIMATION (KDE)

def fit_density(points, bandwidth):
    """Train a Gaussian KDE model using the given points."""

    # Create a Gaussian KDE model with the selected bandwidth
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)

    # Fit the KDE model to the training points
    return kde.fit(points)


# SECTION 11: EVALUATING KDE ON THE GRID

def evaluate_on_grid(kde, grid_points, xx):
    """Calculate the KDE density for every grid cell."""

    # score_samples returns log density
    # np.exp converts the log density back to normal density
    # Reshape the result to match the grid dimensions
    return np.exp(kde.score_samples(grid_points)).reshape(xx.shape)


# SECTION 12: NORMALIZING THE DENSITY VALUES

def normalise_peak(surface):
    """Normalize a density map so its highest value becomes 1."""

    # Find the highest density value
    peak = surface.max()

    # Divide all values by the peak to scale them between 0 and 1
    # Avoid division by zero if the peak is zero
    return surface / peak if peak > 0 else surface


# SECTION 13: QUANTIZATION

def quantize(surface_01):
    """Convert normalized density values into 8-bit integers."""

    # Limit values between 0 and 1
    # Multiply by 255 and convert to uint8 to save memory
    return np.round(np.clip(surface_01, 0.0, 1.0) * 255).astype(np.uint8)


# SECTION 14: DEQUANTIZATION

def dequantize(codes):
    """Convert stored 8-bit values back into values between 0 and 1."""

    # Divide by 255 to recover the approximate normalized values
    return codes.astype(float) / 255.0
