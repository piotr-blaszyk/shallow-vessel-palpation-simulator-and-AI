import pandas as pd

csv_path = "/home/psb120/Documents/diff-tactile-fork/logs/my_experiment/run_20260302_155427/metrics.csv"

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