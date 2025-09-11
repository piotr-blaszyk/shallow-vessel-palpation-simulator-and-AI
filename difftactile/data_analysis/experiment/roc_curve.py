import numpy as np
import matplotlib.pyplot as plt


class Foo:
    def __init__(self):
        pass

    def plot_roc_curve(self):
        fpr_list = np.linspace(0, 1, 11)
        tpr_list = fpr_list ** 0.5
        fpr = np.linspace(0, 1, 101)
        tpr = fpr ** 0.5
        # fpr = fpr_list
        # tpr = tpr_list
        auc = 0.75
        thresholds = np.linspace(0, 1, 11)

        fontsize = 20
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"ROC curve", alpha=0.8, linewidth=6.0)
        plt.scatter(fpr_list, tpr_list, color="red", s=200, label="thresholds")

        for thr, x, y in zip(thresholds, fpr_list, tpr_list):
            plt.text(x, y, f"{thr:.2f}", fontsize=fontsize, ha="left", va="bottom", fontweight="bold")
        
        plt.tick_params(axis="both", which="major", labelsize=fontsize)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight("bold")

        plt.plot([0, 1], [0, 1], "k-", alpha=0.5, linewidth=6.0)
        plt.xlabel("False Positive Rate", fontsize=fontsize, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=fontsize, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.3, linewidth=3.0)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(3.0)
        plt.tight_layout()
        plt.savefig('difftactile/output/roc_curve.pdf', format="pdf", dpi=300)
        plt.show()


def main():
    foo = Foo()
    foo.plot_roc_curve()


if __name__ == '__main__':
    main()
