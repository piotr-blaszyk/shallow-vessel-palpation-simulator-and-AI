def training_step(batch, model, optimizer, lambda_cls=1.0, lambda_reg=1.0):
    # Unpack batch
    x, edge_index, batch_idx, gt_regs_list = batch  # using PyG-style batching
    pred_logits, pred_regs = model(x, edge_index, batch_idx)  
    # pred_logits: [total_slots], pred_regs: [total_slots, 2]
    # Assume model outputs per-graph slots, grouped by batch_idx

    total_cls_loss, total_reg_loss = 0.0, 0.0
    n_graphs = batch_idx.max().item() + 1

    for g in range(n_graphs):
        mask = (batch_idx == g)
        logits_g = pred_logits[mask]
        regs_g = pred_regs[mask]
        gt_regs_g = gt_regs_list[g]

        cls_loss, reg_loss = compute_loss(logits_g, regs_g, gt_regs_g)
        total_cls_loss += cls_loss
        total_reg_loss += reg_loss

    total_loss = lambda_cls * total_cls_loss + lambda_reg * total_reg_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), total_cls_loss.item(), total_reg_loss.item()

def compute_loss(pred_logits, pred_regs, gt_regs, cls_weight=1.0, reg_weight=1.0):
    device = pred_logits.device
    n_slots = pred_logits.size(0)
    n_gt = gt_regs.size(0)

    matched_pred, matched_gt = hungarian_match(pred_logits, pred_regs, gt_regs,
                                               cls_weight=cls_weight,
                                               reg_weight=reg_weight)

    # --- Classification loss ---
    cls_targets = torch.zeros(n_slots, device=device)
    cls_targets[matched_pred] = 1.0
    cls_loss = F.binary_cross_entropy_with_logits(pred_logits, cls_targets, reduction="mean")

    # --- Regression loss ---
    if n_gt > 0:
        matched_pred_regs = pred_regs[matched_pred]      # [n_gt, 2]
        matched_gt_regs = gt_regs[matched_gt].to(device) # [n_gt, 2]
        reg_loss = F.smooth_l1_loss(matched_pred_regs, matched_gt_regs, reduction="mean")
    else:
        reg_loss = torch.tensor(0.0, device=device)

    return cls_loss, reg_loss

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def hungarian_match(pred_logits, pred_regs, gt_regs, cls_weight=1.0, reg_weight=1.0):
    """
    pred_logits: [n_slots]
    pred_regs:   [n_slots, 2]
    gt_regs:     [n_gt, 2]
    returns: matched_pred_indices, matched_gt_indices
    """
    n_slots = pred_regs.size(0)
    n_gt = gt_regs.size(0)

    # Convert logits to probability for cost
    prob = torch.sigmoid(pred_logits)  # [n_slots]

    # Classification cost: want high prob for positives
    cls_cost = -prob.unsqueeze(1).expand(n_slots, n_gt)  # [n_slots, n_gt]

    # Regression cost: L1 distance between predictions and gt
    reg_cost = torch.cdist(pred_regs, gt_regs, p=1)  # [n_slots, n_gt]

    # Weighted sum
    cost_matrix = cls_weight * cls_cost + reg_weight * reg_cost
    cost_matrix = cost_matrix.cpu().detach().numpy()

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)
