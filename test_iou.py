import torch
import numpy as np

def iou_score_from_model(preds, targets, eps=1e-6):
    # This is the implementation from the model
    intersection = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets).sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def iou_score_reference(preds, targets, eps=1e-6):
    # A simpler reference implementation that works on flattened tensors
    preds_flat = preds.reshape(preds.shape[0], preds.shape[1], -1)
    targets_flat = targets.reshape(targets.shape[0], targets.shape[1], -1)
    
    intersection = (preds_flat * targets_flat).sum(dim=2)
    union = (preds_flat + targets_flat).sum(dim=2) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def print_tensor_with_shape(name, tensor):
    print(f"\n{name} (shape: {tensor.shape}):")
    print(tensor)

def run_test_case(test_name, preds, targets):
    print(f"\n=== Test Case: {test_name} ===")
    print_tensor_with_shape("Predictions", preds)
    print_tensor_with_shape("Targets", targets)
    
    iou_model = iou_score_from_model(preds, targets)
    iou_ref = iou_score_reference(preds, targets)
    
    print(f"\nIOU (model implementation): {iou_model:.4f}")
    print(f"IOU (reference implementation): {iou_ref:.4f}")
    print(f"Implementations match: {torch.allclose(iou_model, iou_ref)}")

def main():
    # Test Case 1: Perfect overlap
    perfect_pred = torch.tensor([[[[1., 1.], [1., 1.]]]])  # Shape: [1, 1, 2, 2]
    perfect_target = torch.tensor([[[[1., 1.], [1., 1.]]]])
    run_test_case("Perfect Overlap", perfect_pred, perfect_target)
    
    # Test Case 2: No overlap
    no_overlap_pred = torch.tensor([[[[1., 1.], [1., 1.]]]])
    no_overlap_target = torch.tensor([[[[0., 0.], [0., 0.]]]])
    run_test_case("No Overlap", no_overlap_pred, no_overlap_target)
    
    # Test Case 3: Partial overlap
    partial_pred = torch.tensor([[[[1., 1.], [0., 0.]]]])
    partial_target = torch.tensor([[[[1., 0.], [0., 0.]]]])
    run_test_case("Partial Overlap", partial_pred, partial_target)
    
    # Test Case 4: Multiple frames
    multi_frame_pred = torch.tensor([
        [[[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]]]
    ])  # Shape: [1, 2, 2, 2]
    multi_frame_target = torch.tensor([
        [[[1., 1.], [1., 1.]],
         [[0., 0.], [0., 0.]]]
    ])
    run_test_case("Multiple Frames", multi_frame_pred, multi_frame_target)
    
    # Test Case 5: Multiple batches
    multi_batch_pred = torch.tensor([
        [[[1., 1.], [1., 1.]]],
        [[[0., 0.], [0., 0.]]]
    ])  # Shape: [2, 1, 2, 2]
    multi_batch_target = torch.tensor([
        [[[1., 1.], [1., 1.]]],
        [[[0., 0.], [0., 0.]]]
    ])
    run_test_case("Multiple Batches", multi_batch_pred, multi_batch_target)

if __name__ == "__main__":
    main() 