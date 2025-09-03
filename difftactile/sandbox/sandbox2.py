from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

def evaluate_and_plot_roc(model, dataloader, device="cpu"):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)  # shape: [num_nodes, num_classes]
            probs = torch.softmax(out, dim=-1)[:, 1]  # probability of class=1
            labels = batch.y  # true labels per node
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute AUC
    auc = roc_auc_score(all_labels, all_probs)

    # Compute ROC curve (with sklearn default 100 thresholds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    # Now compute TPR/FPR manually for 20 thresholds
    thresholds = np.linspace(0.0, 1.0, 20)
    tpr_list, fpr_list = [], []
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", alpha=0.8)
    plt.scatter(fpr_list, tpr_list, color="red", s=30, label="20 thresholds")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Node Classification GNN")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    return auc
