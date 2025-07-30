import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from difftactile.cnn.train import *


def visualize_predictions(model_path=None, num_samples=5):
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
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
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
            axes[i, 0].imshow(image, cmap="gray")
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(mask, cmap="gray")
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")
    plt.show()


if __name__ == "__main__":
    visualize_predictions(model_path="saved_models/final_segmentation_model.pt")
