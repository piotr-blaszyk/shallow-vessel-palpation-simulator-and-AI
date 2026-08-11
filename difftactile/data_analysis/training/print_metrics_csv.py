import glob
import os
import sys

import pandas as pd

from difftactile.main.paths import repo_path

# Training writes one timestamped run directory per invocation under
# logs/my_experiment/. Default to the most recent run so this works on any
# machine; pass an explicit metrics.csv path to inspect an older run.
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    candidates = sorted(glob.glob(repo_path("logs/my_experiment/run_*/metrics.csv")))
    if not candidates:
        raise FileNotFoundError(
            "No logs/my_experiment/run_*/metrics.csv found. Train a model first, "
            "or pass a metrics.csv path as the first argument."
        )
    csv_path = candidates[-1]

print(f"Reading metrics from: {os.path.relpath(csv_path, repo_path())}")
df = pd.read_csv(csv_path)

columns = ["epoch", "val/loss", "val_iou/0", "val_iou/1"]

missing = [col for col in columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

filtered_df = df.dropna(subset=["val/loss", "val_iou/0", "val_iou/1"])
result = filtered_df[columns].copy()
result["epoch"] = result["epoch"].astype(int)
result = result.set_index("epoch")
result = result.sort_index()

formatters = {
    "val/loss": "{:.4f}".format,
    "val_iou/0": "{:.2f}".format,
    "val_iou/1": "{:.2f}".format,
}

print("\nValidation Metrics per Epoch:\n")
print(result.to_string(formatters=formatters))