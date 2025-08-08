import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from difftactile.cnn.train import *
import matplotlib.colors as mcolors


def calculate_iou(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou_score


def create_confusion_matrix_overlay(ground_truth, prediction):
    overlay = np.zeros((*ground_truth.shape, 3))
    
    # True Negative (black)
    tn_mask = (ground_truth == 0) & (prediction == 0)
    overlay[tn_mask] = [0, 0, 0]
    
    # True Positive (white)
    tp_mask = (ground_truth == 1) & (prediction == 1)
    overlay[tp_mask] = [1, 1, 1]
    
    # False Positive (red)
    fp_mask = (ground_truth == 0) & (prediction == 1)
    overlay[fp_mask] = [1, 0, 0]
    
    # False Negative (blue)
    fn_mask = (ground_truth == 1) & (prediction == 0)
    overlay[fn_mask] = [0, 0, 1]
    
    return overlay


def visualize_predictions(model_path, num_samples):
    full_dataset = MyDataset(
        data_dir=SYSTEM_PARAMS.files.dataset_root
    )
    train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
        full_dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = SegmentationModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    plt.tight_layout(pad=3.0)

    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            if i >= num_samples:
                break
            image = image.to(device)
            logits = model(image)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            
            image = image.cpu().numpy().squeeze()
            mask = mask.cpu().numpy().squeeze()
            pred = pred.cpu().numpy().squeeze()
            
            iou_score = calculate_iou(mask, pred)
            
            confusion_overlay = create_confusion_matrix_overlay(mask, pred)
            
            axes[i, 0].imshow(image, cmap="gray")
            axes[i, 0].set_title(f"sample {i+1}")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(confusion_overlay)
            axes[i, 1].set_title(f"IoU: {iou_score:.3f}")
            axes[i, 1].axis("off")
            
    plt.show()


if __name__ == "__main__":
    visualize_predictions(
        model_path=SYSTEM_PARAMS.files.segmentation_model_weights,
        num_samples=5
    )
