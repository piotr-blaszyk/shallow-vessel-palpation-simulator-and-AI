import json
import numpy as np
import matplotlib.pyplot as plt
import csv


class DomainAdaptationLineChart:
    def __init__(self):
        self.targets = "difftactile/output/bo_all_targets.json"
        self.params = "difftactile/output/bo_all_params.json"
        self.csv_output = "difftactile/output/bo_merged.csv"

    def generate_line_chart(self):
        # --- Load data ---
        with open(self.targets, "r") as f:
            targets = json.load(f)  # list of floats, length n
        with open(self.params, "r") as f:
            params = json.load(f)   # list of dicts, length n

        # --- Build numpy array ---
        param_keys = list(params[0].keys())  # get the k keys
        n = len(targets)
        k = len(param_keys)

        # Column names: ["target"] + param_keys
        col_names = ["target"] + param_keys

        # Create numpy array of shape (n, 1+k)
        data = np.zeros((n, 1 + k))
        data[:, 0] = np.array(targets)
        for j, key in enumerate(param_keys):
            data[:, j + 1] = np.array([p[key] for p in params])

        # --- Min-max normalization ---
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_norm = (data - data_min) / (data_max - data_min + 1e-12)

        # --- Plot ---
        plt.figure(figsize=(10, 6))

        for j, col in enumerate(col_names):
            if col == "target":
                plt.plot(
                    range(n),
                    data_norm[:, j],
                    label=col,
                    linewidth=3.0,
                    color="red"
                )
            else:
                plt.plot(
                    range(n),
                    data_norm[:, j],
                    label=col,
                    linewidth=1.5
                )

        plt.xlabel("Bayesian Optimisation step number")
        plt.ylabel("Target and parameter values")
        plt.title("Normalised Targets and Parameters over BO Steps")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("difftactile/output/domain_adaptation_line_chart.pdf", format="pdf", dpi=300)
        plt.show()

        return data, col_names, data_norm

    def merge_clean_write(self):
        # --- Load data ---
        with open(self.targets, "r") as f:
            targets = json.load(f)
        with open(self.params, "r") as f:
            params = json.load(f)

        # --- Merge into a list of dicts ---
        merged = []
        for t, p in zip(targets, params):
            row = {"target": t}
            row.update(p)
            merged.append(row)

        # --- Write to CSV with scientific notation ---
        fieldnames = ["target"] + list(params[0].keys())
        with open(self.csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged:
                formatted_row = {
                    key: f"{row[key]:.1e}" for key in fieldnames
                }
                writer.writerow(formatted_row)


if __name__ == '__main__':
    chart = DomainAdaptationLineChart()
    chart.merge_clean_write()
