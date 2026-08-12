from test_fingerspell import (
    ASLNOW_ROOT,
    TARGET_ORIENTATION,
    get_orientation
)

import os
import json
import numpy as np


folder = os.path.join(
    ASLNOW_ROOT,
    "J"
)

count = 0

for filename in sorted(os.listdir(folder)):

    if not filename.endswith(".json"):
        continue

    path = os.path.join(
        folder,
        filename
    )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = np.array(
            [[p["x"], p["y"]] for p in data],
            dtype=np.float32
        )

        if points.shape != (21, 2):
            continue

        if get_orientation(points) != TARGET_ORIENTATION:
            continue

        print(filename)

        count += 1

        if count >= 10:
            break

    except Exception:
        continue

print()
print("Compatible J samples:", count)