import json
import numpy as np
import matplotlib.pyplot as plt
import csv


class DomainAdaptationLineChart:
    def __init__(self):
        self.targets = "difftactile/output/bo_all_targets.json"
        self.params = "difftactile/output/bo_all_params_renamed.json"
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

        # Column names: ["MAE"] + param_keys
        col_names = ["MAE"] + param_keys

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
        plt.figure(figsize=(10, 5))

        for j, col in enumerate(col_names):
            if col == "MAE":
                plt.plot(
                    range(n),
                    data_norm[:, j],
                    label=col,
                    linewidth=12.0,
                    color="red"
                )
            else:
                plt.plot(
                    range(n),
                    data_norm[:, j],
                    label=col,
                    linewidth=6.0
                )

        fontsize = 20
        # Axis labels with larger, bold font
        plt.xlabel("Bayesian Optimisation step number", fontsize=fontsize, fontweight="bold")
        plt.ylabel("Normalised target\nand parameter values", fontsize=fontsize, fontweight="bold")

        # Tick labels with larger, bold font
        plt.tick_params(axis="both", which="major", labelsize=fontsize)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight("bold")

        # Legend with larger, bold font
        # plt.legend(fontsize=1000, prop={"weight": "bold"})
        # plt.legend(prop={"size": 16, "weight": "bold"})
        plt.legend(
            prop={"size": 16, "weight": "bold"},
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0
        )

        plt.grid(True, linestyle="--", alpha=0.6, linewidth=6.0)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(6.0)
        # plt.tight_layout(rect=[0, 0, 0.85, 1])  # shrink plot to make room for legend
        plt.subplots_adjust(right=0.8)
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
    chart.generate_line_chart()
