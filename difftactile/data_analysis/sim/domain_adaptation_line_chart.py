import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import cv2
from scipy.optimize import linear_sum_assignment
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


class DomainAdaptationLineChart:
    def __init__(self):
        self.targets = "difftactile/output/bo_all_targets.json"
        self.params = "difftactile/output/bo_all_params_renamed.json"
        self.csv_output = "difftactile/output/bo_merged.csv"

    def generate_line_chart(self):
        with open(self.targets, "r") as f:
            targets = json.load(f)  # list of floats, length n
        with open(self.params, "r") as f:
            params = json.load(f)   # list of dicts, length n
        
        param_keys = list(params[0].keys())  # get the k keys
        n = len(targets)
        k = len(param_keys)
        col_names = ["mean average error"] + param_keys
        data = np.zeros((n, 1 + k))
        data[:, 0] = np.array(targets)
        for j, key in enumerate(param_keys):
            data[:, j + 1] = np.array([p[key] for p in params])

        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_norm = (data - data_min) / (data_max - data_min + 1e-12)
        plt.figure(figsize=(10, 5))

        for j, col in enumerate(col_names):
            if col == "mean average error":
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
        plt.xlabel("Bayesian Optimisation step number", fontsize=fontsize, fontweight="bold")
        plt.ylabel("Normalised target\nand parameter values", fontsize=fontsize, fontweight="bold")
        plt.tick_params(axis="both", which="major", labelsize=fontsize)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight("bold")
        plt.legend(
            prop={"size": 16, "weight": "bold"},
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0
        )
        
        plt.grid(True, linestyle="--", alpha=0.6, linewidth=6.0)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(6.0)
        plt.subplots_adjust(right=0.8)
        plt.tight_layout()
        plt.savefig("difftactile/output/domain_adaptation_line_chart.pdf", format="pdf", dpi=300)
        plt.show()

        return data, col_names, data_norm

    def generate_ridgeline_chart(self):
        # Load data
        with open(self.targets, "r") as f:
            targets = json.load(f)  # list of floats, length n
        with open(self.params, "r") as f:
            params = json.load(f)   # list of dicts, length n

        # Build dataframe
        param_keys = list(params[0].keys())  # get parameter names
        n = len(targets)
        k = len(param_keys)
        col_names = ["mean average error"] + param_keys

        data = np.zeros((n, 1 + k))
        data[:, 0] = np.array(targets)
        for j, key in enumerate(param_keys):
            data[:, j + 1] = np.array([p[key] for p in params])

        df = pd.DataFrame(data, columns=col_names)

        # Normalize each column independently
        df_norm = (df - df.min()) / (df.max() - df.min() + 1e-12)

        # Ridgeline plot
        plt.figure(figsize=(12, 8))
        offset = 1.2  # vertical spacing between ridgelines

        x = np.arange(n)
        for i, col in enumerate(col_names):
            y = df_norm[col] + i * offset
            if col == "mean average error":
                plt.plot(x, y, color="red", linewidth=2.5, label=col, zorder=3)
                plt.fill_between(x, i * offset, y, color="red", alpha=0.25)
            else:
                plt.plot(x, y, linewidth=1.5, label=col, zorder=2)
                plt.fill_between(x, i * offset, y, alpha=0.25)

        # Formatting
        fontsize = 20
        plt.xlabel("Bayesian Optimisation step number", fontsize=fontsize, fontweight="bold")
        plt.ylabel("Normalised value", fontsize=fontsize, fontweight="bold")
        plt.yticks(
            [i * offset for i in range(len(col_names))],
            col_names,
            fontsize=fontsize,
            fontweight="bold"
        )
        plt.xticks(fontsize=fontsize, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.4)
        # plt.title("Ridgeline of Normalised Target and Parameters", fontsize=16, fontweight="bold")

        plt.tight_layout()
        plt.savefig("difftactile/output/domain_adaptation_ridgeline_chart.pdf", format="pdf", dpi=300)
        plt.show()

        return df, col_names, df_norm

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

    def post_process_images(self):
        img = cv2.imread("difftactile/output/da_overlay_twist_z.png")
        b, g, r = cv2.split(img)

        _, mask_green = cv2.threshold(g, 250, 255, cv2.THRESH_BINARY)
        _, mask_red = cv2.threshold(r, 250, 255, cv2.THRESH_BINARY)

        def get_positions(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            positions = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    positions.append((cx, cy))
            return positions

        green_positions = get_positions(mask_green)
        red_positions = get_positions(mask_red)

        cost_matrix = np.zeros((len(green_positions), len(red_positions)), dtype=np.float32)
        for i, (gx, gy) in enumerate(green_positions):
            for j, (rx, ry) in enumerate(red_positions):
                cost_matrix[i, j] = (gx - rx) ** 2 + (gy - ry) ** 2

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        h, w = img.shape[:2]
        output = np.ones((h, w, 3), dtype=np.uint8) * 255

        for (x, y) in green_positions:
            cv2.circle(output, (x, y), 10, (0, 255, 0), -1)
        for (x, y) in red_positions:
            cv2.circle(output, (x, y), 10, (0, 0, 255), -1)

        for g_idx, r_idx in zip(row_ind, col_ind):
            gx, gy = green_positions[g_idx]
            rx, ry = red_positions[r_idx]
            cv2.line(output, (gx, gy), (rx, ry), (255, 0, 0), 6)

        cv2.imwrite('difftactile/output/twist_z_2.png', output)
        cv2.imshow("Redrawn Output with Matches", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    chart = DomainAdaptationLineChart()
    chart.generate_ridgeline_chart()
