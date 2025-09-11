import json
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
from scipy.optimize import linear_sum_assignment

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
    
    def post_process_images(self):
        # Load the image
        img = cv2.imread("difftactile/output/da_overlay_press.png")

        # Split into channels
        b, g, r = cv2.split(img)

        # Threshold the green channel
        _, mask = cv2.threshold(g, 250, 255, cv2.THRESH_BINARY)

        # Clean up small noise using morphology
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours of the green dots
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        positions = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:  # avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                positions.append((cx, cy))

                # Draw detected centers on the original image
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

        # Print results
        print("Detected positions:", positions)

        # Show the results
        cv2.imshow("Mask", mask)
        cv2.imshow("Detected Dots", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def post_process_images_2(self):
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
    chart.post_process_images_2()
