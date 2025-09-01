from difftactile.cnn.gnn import *


def choose_optimal_threshold():
    BATCH_SIZE = 512
    NUM_WORKERS = 16
    VAL_EPOCH_SUBSET_SIZE = BATCH_SIZE * 8
    with open(SYSTEM_PARAMS.files.test_loader_gnn, "rb") as f:
        test_data = pickle.load(f)
    test_dataset = test_data["dataset"]
    test_dataset.set_difficulty_level(1.0)
    data_module = MyDataModule(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
        train_subset_size=VAL_EPOCH_SUBSET_SIZE,
        val_subset_size=VAL_EPOCH_SUBSET_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    model = GNN(lr=-1)
    model.load_state_dict(torch.load(SYSTEM_PARAMS.files.final_segmentation_model_gnn))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    thresholds = np.linspace(0.4, 0.6, 5)
    best_threshold = 0.5
    best_fg_iou = 0.0
    print("\nTesting different thresholds:")
    print("Threshold | Foreground IoU")
    print("-" * 25)
    val_loader = data_module.val_dataloader()
    with torch.no_grad():
        for threshold in thresholds:
            total_fg_iou = 0.0
            num_batches = 0
            for batch, _ in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                logits = logits.squeeze(-1)
                mask = batch.mask
                logits = logits[mask]
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                iou_scores = GNN.iou_score(preds, batch.y)
                total_fg_iou += iou_scores["fg_iou"].item()
                num_batches += 1
            avg_fg_iou = total_fg_iou / num_batches
            print(f"{threshold:.2f}     | {avg_fg_iou:.4f}")
            if avg_fg_iou > best_fg_iou:
                best_fg_iou = avg_fg_iou
                best_threshold = threshold
    print("\nBest results:")
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Best foreground IoU: {best_fg_iou:.4f}")
