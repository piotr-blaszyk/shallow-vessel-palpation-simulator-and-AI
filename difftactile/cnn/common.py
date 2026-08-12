import torch
import torch.nn.functional as F


# Key stamped into a test-loader pickle to record that it came from the meat
# ("C") scheme, whose `dataset_stats` entry is a single flat stats dict instead
# of the curriculum's {difficulty: stats} mapping.
MEAT_LOADER_FLAG = "meat"
# The published Zenodo bundle predates the rename and stamps the same flag under
# the old conference-derived key, so both spellings are accepted on read.
LEGACY_MEAT_LOADER_FLAG = "iros"


def has_flat_stats(test_data):
    """True when `test_data['dataset_stats']` is a single stats dict.

    Meat-scheme loaders store stats flat; simulation/curriculum loaders store
    them keyed by difficulty. Accepts the pre-rename key so that the published
    bundle's pickles keep loading unchanged.
    """
    return MEAT_LOADER_FLAG in test_data or LEGACY_MEAT_LOADER_FLAG in test_data


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.9, beta=0.1, smooth=1e-6):
        """
        Tversky loss for imbalanced data
        :param alpha: weight of false negatives (default: 0.7)
        :param beta: weight of false positives (default: 0.3)
        :param smooth: smoothing constant
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Sigmoid activation
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        
        return 1 - tversky


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        """Focal Loss for binary classification.
        
        Args:
            alpha (float): Weight factor for the rare class, range [0,1].
                         - alpha=0.5: Equal weight for both classes (default)
                         - alpha>0.5: More weight on rare/positive class
                         - alpha<0.5: More weight on common/negative class
            gamma (float): Focusing parameter, range [0,∞).
                         - gamma=0.0: Equivalent to BCE loss (default)
                         - gamma=2.0: Typical value from paper
                         - Higher gamma reduces relative loss for well-classified examples,
                           putting more focus on hard, misclassified examples
        
        Reference:
            "Focal Loss for Dense Object Detection" https://arxiv.org/abs/1708.02002
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class Common:
    @staticmethod
    def iou_score(preds, targets, eps=1e-6):
        """
        Compute class-wise IoU scores for binary segmentation.
        
        Args:
            preds: Binary predictions (B, T, H, W)
            targets: Binary ground truth (B, T, H, W)
            eps: Small constant to avoid division by zero
            
        Returns:
            Dictionary containing various IoU metrics
        """
        # Compute frame-wise ground truth presence (B, T)
        gt_presence = targets.sum(dim=(2, 3)) > 0
        pred_presence = preds.sum(dim=(2, 3)) > 0
        
        # Split frames based on ground truth presence
        has_gt_mask = gt_presence
        no_gt_mask = ~gt_presence
        
        # Initialize metrics
        metrics = {
            'fg_iou': torch.tensor(0.0, device=preds.device),
            'bg_iou': torch.tensor(0.0, device=preds.device),
            'macro_iou': torch.tensor(0.0, device=preds.device),
            'detection_rate': torch.tensor(0.0, device=preds.device)
        }
        
        # Compute IoU for frames with foreground presence
        if has_gt_mask.sum() > 0:
            # Foreground IoU (class 1)
            fg_intersection = (preds[has_gt_mask] * targets[has_gt_mask]).sum(dim=(1, 2))
            fg_union = (preds[has_gt_mask] + targets[has_gt_mask]).sum(dim=(1, 2)) - fg_intersection
            fg_iou = (fg_intersection + eps) / (fg_union + eps)
            metrics['fg_iou'] = fg_iou.mean()
            
            # Background IoU (class 0)
            bg_preds = 1 - preds[has_gt_mask]
            bg_targets = 1 - targets[has_gt_mask]
            bg_intersection = (bg_preds * bg_targets).sum(dim=(1, 2))
            bg_union = (bg_preds + bg_targets).sum(dim=(1, 2)) - bg_intersection
            bg_iou = (bg_intersection + eps) / (bg_union + eps)
            metrics['bg_iou'] = bg_iou.mean()
            
            # Macro IoU (mean of class IoUs)
            metrics['macro_iou'] = (metrics['fg_iou'] + metrics['bg_iou']) / 2
        
        # Compute detection rate for empty frames
        if no_gt_mask.sum() > 0:
            # For empty GT frames: IoU=1 if prediction also empty, 0 if not empty
            correct_empty = (~pred_presence[no_gt_mask]).float()
            metrics['detection_rate'] = correct_empty.mean()
        
        return metrics
