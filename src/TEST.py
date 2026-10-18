# This code retrieves the minimum and maximum UV and log-scaled IR values
# from the training dataset. These values are needed to normalize the
# ESP32 sensor readings using the same scaling as the trained KDE model,
# allowing the model to classify light sources correctly.

from pipeline_common import load_split_scaled

data = load_split_scaled()

lo, hi = data["bounds"]

print("UV_TRAIN_MIN =", lo[0])
print("UV_TRAIN_MAX =", hi[0])
print("LOG_IR_TRAIN_MIN =", lo[1])
print("LOG_IR_TRAIN_MAX =", hi[1])