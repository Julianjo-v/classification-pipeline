"""This script converts a set of precomputed density maps stored in surfaces.npz
into a C header file called model.h for an Arduino or ESP32 project. It loads one
square grid per class, checks that each grid is the expected size and that its values
fit in 0 to 255, then writes them out as uint8_t lookup tables stored in flash memory.
It also pulls class names from the training data and labels, and adds a small helper function,
modelRead, that safely looks up a value by class, row, and column. Run with --check, it skips
writing and just confirms that the existing model.h matches what the script would generate."""

import argparse
from pathlib import Path

import numpy as np

from pipeline_common import GRID_N, cluster_names, load_split_scaled


HERE = Path(__file__).resolve().parent
SURFACES_PATH = HERE / "surfaces.npz"
LABELS_PATH = HERE / "labels.csv"
OUTPUT_PATH = HERE / "model.h"


def render_header():
    with np.load(SURFACES_PATH) as archive:
        class_ids = sorted(int(key[1:]) for key in archive.files)
        if class_ids != list(range(len(class_ids))):
            raise ValueError(f"surface IDs must be contiguous from zero: {class_ids}")
        tables = []
        for class_id in class_ids:
            table = np.asarray(archive[f"c{class_id}"])
            if table.shape != (GRID_N, GRID_N):
                raise ValueError(
                    f"c{class_id} has shape {table.shape}; expected {(GRID_N, GRID_N)}"
                )
            if not np.issubdtype(table.dtype, np.integer):
                table = np.rint(np.clip(table, 0.0, 1.0) * 255.0)
            if np.any((table < 0) | (table > 255)):
                raise ValueError(f"c{class_id} contains values outside 0..255")
            tables.append(table.astype(np.uint8))

    split = load_split_scaled()
    lo, hi = split["bounds"]
    if not np.all(np.isfinite([*lo, *hi])):
        raise ValueError("training scale bounds contain a non-finite value")
    labels = np.loadtxt(LABELS_PATH, dtype=int)
    if len(labels) != len(split["train"]):
        raise ValueError("labels.csv length does not match the training split")
    names_by_id = cluster_names(labels, split["y_train"])
    class_names = [names_by_id[class_id].upper() for class_id in class_ids]

    lines = [
        "#ifndef MODEL_H",
        "#define MODEL_H",
        "",
        "#include <Arduino.h>",
        "#include <stdint.h>",
        "#include <pgmspace.h>",
        "",
        f"// Generated from surfaces.npz: {GRID_N}x{GRID_N} peak-normalized KDE maps.",
        "// Values are uint8_t (0..255); normalized density = value / 255.0f.",
        f"// Grid coordinates: x = column a/ {GRID_N - 1}.0f, y = row / {GRID_N - 1}.0f.",
        "// These are density lookup tables, not a trained classifier by themselves.",
        "",
        f"static constexpr uint8_t MODEL_GRID_SIZE = {GRID_N};",
        f"static constexpr uint8_t MODEL_NUM_CLASSES = {len(class_ids)};",
        "static const char* const MODEL_CLASS_IDS[MODEL_NUM_CLASSES] = {",
    ]
    lines.extend(
        f'    "{name}"{"," if index + 1 < len(class_names) else ""}'
        for index, name in enumerate(class_names)
    )
    lines.extend(["};", ""])

    for class_id, table in zip(class_ids, tables):
        lines.append(
            f"static const uint8_t MODEL_CLASS_{class_id}[{GRID_N}][{GRID_N}] PROGMEM = {{"
        )
        for row in table:
            values = ", ".join(f"{int(value):3d}" for value in row)
            lines.append(f"    {{{values}}},")
        lines.extend(["};", ""])

    lines.append("static const uint8_t* const MODEL_SURFACES[MODEL_NUM_CLASSES] = {")
    lines.extend(
        f"    &MODEL_CLASS_{class_id}[0][0],"
        for class_id in class_ids
    )
    lines.extend(
        [
            "};",
            "",
            f"// class_index is 0..MODEL_NUM_CLASSES-1; row/col are 0..{GRID_N - 1}.",
            "inline uint8_t modelRead(uint8_t class_index, uint8_t row, uint8_t col) {",
            "    if (class_index >= MODEL_NUM_CLASSES || row >= MODEL_GRID_SIZE || col >= MODEL_GRID_SIZE) return 0;",
            "    return pgm_read_byte(MODEL_SURFACES[class_index] + row * MODEL_GRID_SIZE + col);",
            "}",
            "",
            "#endif // MODEL_H",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render_header()
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != expected:
            raise SystemExit("model.h does not match generated output")
        print("model.h exactly matches generated output")
        return
    OUTPUT_PATH.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
