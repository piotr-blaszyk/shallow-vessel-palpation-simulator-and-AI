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
    full_dataset = SegmentationDataset(
        SYSTEM_PARAMS.files.training_data_markers_folder,
        SYSTEM_PARAMS.files.training_data_segmentation_mask_folder,
    )
    _, _, test_dataset = SegmentationDataset.create_splits(full_dataset)
    test_transforms = A.Compose([ToTensorV2()])
    test_dataset = TransformDataset(test_dataset, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = SegmentationModel()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    else:
        checkpoint_dir = "lightning_logs"
        versions = [d for d in os.listdir(checkpoint_dir) if d.startswith("version_")]
        if versions:
            latest_version = max(versions, key=lambda x: int(x.split("_")[1]))
            checkpoints_dir = os.path.join(
                checkpoint_dir, latest_version, "checkpoints"
            )
            checkpoint_files = os.listdir(checkpoints_dir)
            if checkpoint_files:
                model_path = os.path.join(checkpoints_dir, checkpoint_files[0])
                model.load_state_dict(torch.load(model_path))
                print(f"Loaded model from {model_path}")
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
            pred = model(image)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            
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
