from pipeline_common import load_split_scaled

data = load_split_scaled()

lo, hi = data["bounds"]

print("UV_TRAIN_MIN =", lo[0])
print("UV_TRAIN_MAX =", hi[0])
print("LOG_IR_TRAIN_MIN =", lo[1])
print("LOG_IR_TRAIN_MAX =", hi[1])