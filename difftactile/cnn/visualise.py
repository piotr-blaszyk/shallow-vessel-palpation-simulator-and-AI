import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from difftactile.cnn.train import *
import matplotlib.colors as mcolors
import pickle
import cv2


class Visualisation:
    def __init__(self):
        pass

    def calculate_iou(self, ground_truth, prediction):
        intersection = np.logical_and(ground_truth, prediction)
        union = np.logical_or(ground_truth, prediction)
        iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        return iou_score

    def create_confusion_matrix_overlay(self, ground_truth, prediction):
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

    def visualize_predictions(self):
        model_path = SYSTEM_PARAMS.files.segmentation_model_weights
        num_samples = 5
        with open(SYSTEM_PARAMS.files.test_loader, 'rb') as f:
            test_data = pickle.load(f)
        test_loader = DataLoader(
            test_data['dataset'],
            batch_size=1,
            shuffle=True,
            num_workers=test_data['num_workers']
        )

        if False:
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

                    n = image.shape[0]
                    image = image[n // 2, :, :]
                    mask = mask[n // 2, :, :]
                    pred = pred[n // 2, :, :]
                    
                    iou_score = self.calculate_iou(mask, pred)
                    
                    confusion_overlay = self.create_confusion_matrix_overlay(mask, pred)
                    
                    axes[i, 0].imshow(image, cmap="gray")
                    axes[i, 0].set_title(f"sample {i+1}")
                    axes[i, 0].axis("off")
                    
                    axes[i, 1].imshow(confusion_overlay)
                    axes[i, 1].set_title(f"IoU: {iou_score:.3f}")
                    axes[i, 1].axis("off")
                    
            plt.show()

    def visualize_dataset(self):
        BATCH_SIZE = 1
        NUM_WORKERS = 1
        full_dataset = MyDataset(
            data_dir=SYSTEM_PARAMS.files.dataset_root
        )
        train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
            full_dataset, train_size=0.70, val_size=0.15, test_size=0.15
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        data_loader = train_loader
        data_iter = iter(data_loader)
        i = 0
        while True:  # Main loop for continuous data loading
            try:
                image, label = next(data_iter)
            except StopIteration:
                print("End of dataset reached. Restarting...")
                data_iter = iter(data_loader)
                continue

            # Convert tensors to numpy arrays
            image_seq = image.numpy().squeeze()  # Shape: (T, H, W)
            label_seq = label.numpy().squeeze()  # Shape: (T, H, W)

            current_frame = 0
            total_frames = image_seq.shape[0]

            while True:
                # Prepare the current frame
                current_image = image_seq[current_frame]
                current_label = label_seq[current_frame]
                
                # Normalize images for display (they are already in range [0,1])
                current_image = (current_image * 255).astype(np.uint8)
                current_label = (current_label * 255).astype(np.uint8)

                # Scale up images by 4x using NEAREST neighbor interpolation
                scale_factor = 4
                current_image = cv2.resize(current_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                current_label = cv2.resize(current_label, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

                # Add frame counter text
                frame_text = f"Frame: {current_frame + 1}/{total_frames}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_color = (255, 255, 255)  # White text
                
                # Get text size to position it at the bottom
                text_size = cv2.getTextSize(frame_text, font, font_scale, font_thickness)[0]
                text_x = 10
                text_y = current_image.shape[0] - 20  # 20 pixels from bottom
                
                # Add black background for text visibility
                padding = 5
                cv2.rectangle(current_image, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                cv2.rectangle(current_label, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                
                # Add text to both images
                cv2.putText(current_image, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                cv2.putText(current_label, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

                # Create display windows
                cv2.imshow(f'Input Image {i}', current_image)
                cv2.imshow(f'Ground Truth Label {i}', current_label)

                # Handle keyboard input
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):  # Quit visualization
                    cv2.destroyAllWindows()
                    return
                elif key == ord('k'):  # Next frame
                    current_frame = (current_frame + 1) % total_frames
                elif key == ord('j'):  # Previous frame
                    current_frame = (current_frame - 1) % total_frames
                elif key == ord('c'):  # Close current sequence and load next
                    i += 1
                    cv2.destroyAllWindows()
                    break

def main():
    v = Visualisation()
    v.visualize_dataset()


if __name__ == "__main__":
    main()
