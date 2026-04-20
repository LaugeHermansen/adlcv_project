import sys, os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import numpy as np
import wandb
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
from tqdm import tqdm

from src.hidden_objects_dataset import HiddenObjectsImageClassLevelFast

# --- Config ---
SPLIT      = "train"
IMAGE_SIZE = 512
BATCH_SIZE = 16
MAX_GT     = 50
LR         = 1e-4
EPOCHS     = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH  = "output/detr_placement.pt"
N_HEATMAPS = 10
RECALL_K   = 5      # top-K predictions used for recall
IOU_THRESH = 0.5    # IoU threshold for a "hit"


def build_model(num_classes):
    detr = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50"
    )

    # Freeze backbone and encoder
    for param in detr.model.backbone.parameters():
        param.requires_grad = False
    for param in detr.model.encoder.parameters():
        param.requires_grad = False

    hidden_dim = detr.config.d_model    # 256

    # Scalar placement score head
    detr.score_head = nn.Linear(hidden_dim, 1)

    # Class conditioning: one embedding per object class, broadcast over 100 queries
    detr.class_embed = nn.Embedding(num_classes, hidden_dim)

    return detr


def make_collate_fn(class_to_idx):
    def collate_fn(batch):
        images = torch.stack([s["image"] for s in batch])   # (B, 3, H, W)

        class_indices = torch.tensor(
            [class_to_idx.get(s['class'], 0) for s in batch],
            dtype=torch.long,
        )

        targets = []
        for s in batch:
            pos_mask = s['labels'].bool()
            boxes = s['boxes'][pos_mask]                    # pixel xywh
            confidences = s['confidences'][pos_mask]
            rewards = s['image_reward_scores'][pos_mask]

            r_min, r_max = rewards.min(), rewards.max()
            norm_rewards = (
                torch.ones_like(rewards) if r_min == r_max
                else (rewards - r_min) / (r_max - r_min)
            )
            weights = confidences * norm_rewards            # (K,)

            if len(weights) > MAX_GT:
                _, keep = weights.topk(MAX_GT)
                boxes, weights = boxes[keep], weights[keep]

            x, y, w, h = boxes.unbind(-1)
            cx = (x + 0.5*w) / IMAGE_SIZE
            cy = (y + 0.5*h) / IMAGE_SIZE
            wn = w / IMAGE_SIZE
            hn = h / IMAGE_SIZE
            boxes_norm = torch.stack([cx, cy, wn, hn], dim=-1).clamp(0, 1)

            targets.append({"boxes": boxes_norm, "weights": weights})

        return images, class_indices, targets
    return collate_fn


def hungarian_match(pred_boxes, gt_boxes):
    cost = torch.cdist(pred_boxes, gt_boxes, p=1).cpu().numpy()
    pred_idx, gt_idx = linear_sum_assignment(cost)
    return pred_idx, gt_idx


def compute_loss(pred_boxes, pred_scores, targets):
    total_box_loss   = 0.0
    total_score_loss = 0.0

    for b in range(len(targets)):
        gt_boxes   = targets[b]['boxes'].to(pred_boxes.device)
        gt_weights = targets[b]['weights'].to(pred_boxes.device)

        pb = pred_boxes[b]      # (100, 4)
        ps = pred_scores[b]     # (100,)

        score_targets = torch.zeros(100, device=pb.device)

        if len(gt_boxes) > 0:
            pred_idx, gt_idx = hungarian_match(pb.detach(), gt_boxes)
            total_box_loss  += nn.functional.l1_loss(pb[pred_idx], gt_boxes[gt_idx])
            score_targets[pred_idx] = gt_weights[gt_idx].float()

        total_score_loss += nn.functional.mse_loss(ps, score_targets)

    n = len(targets)
    return total_box_loss / n, total_score_loss / n


def box_iou_cxcywh(boxes_a, boxes_b):
    """
    boxes_a: (N, 4) normalised cxcywh
    boxes_b: (M, 4) normalised cxcywh
    returns: (N, M) IoU matrix
    """
    # convert to x1y1x2y2
    def to_xyxy(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

    a = to_xyxy(boxes_a)   # (N, 4)
    b = to_xyxy(boxes_b)   # (M, 4)

    inter_x1 = torch.max(a[:, None, 0], b[None, :, 0])
    inter_y1 = torch.max(a[:, None, 1], b[None, :, 1])
    inter_x2 = torch.min(a[:, None, 2], b[None, :, 2])
    inter_y2 = torch.min(a[:, None, 3], b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    area_a  = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])   # (N,)
    area_b  = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])   # (M,)
    union   = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-6)


def forward(model, images, class_indices):
    outputs    = model.model(pixel_values=images)
    seq_out    = outputs.last_hidden_state          # (B, 100, 256)

    class_emb  = model.class_embed(class_indices)   # (B, 256)
    seq_out    = seq_out + class_emb.unsqueeze(1)   # (B, 100, 256)

    pred_boxes  = model.bbox_predictor(seq_out).sigmoid()          # (B, 100, 4)
    pred_scores = model.score_head(seq_out).squeeze(-1).sigmoid()  # (B, 100)
    return pred_boxes, pred_scores


@torch.no_grad()
def evaluate(model, val_dl, desc="Val"):
    """
    Returns dict with:
      val_loss, val_box_loss, val_score_loss  — same losses as training
      recall@K                                — fraction of GT boxes hit by top-K preds (IoU >= IOU_THRESH)
      score_spearman                          — rank correlation of predicted scores vs GT weights
    """
    model.eval()
    total_box   = 0.0
    total_score = 0.0
    n_batches   = 0

    all_recall       = []
    all_pred_scores  = []   # for spearman: predicted score of matched pred
    all_gt_weights   = []   # for spearman: GT weight of matched GT

    for images, class_indices, targets in tqdm(val_dl, desc=desc, leave=False):
        images        = images.to(DEVICE)
        class_indices = class_indices.to(DEVICE)

        pred_boxes, pred_scores = forward(model, images, class_indices)

        box_loss, score_loss = compute_loss(pred_boxes, pred_scores, targets)
        total_box   += box_loss.item()
        total_score += score_loss.item()
        n_batches   += 1

        # Per-image recall and score correlation
        for b in range(len(targets)):
            gt_boxes   = targets[b]['boxes'].to(DEVICE)   # (K, 4)
            gt_weights = targets[b]['weights'].to(DEVICE) # (K,)

            if len(gt_boxes) == 0:
                continue

            pb = pred_boxes[b]    # (100, 4)
            ps = pred_scores[b]   # (100,)

            # Top-K predicted boxes by score
            topk_idx = ps.topk(RECALL_K).indices          # (K,)
            topk_boxes = pb[topk_idx]                     # (K, 4)

            # IoU between top-K preds and all GT boxes
            iou = box_iou_cxcywh(topk_boxes, gt_boxes)   # (K, n_gt)

            # A GT box is "recalled" if any top-K pred has IoU >= threshold
            gt_recalled = (iou.max(dim=0).values >= IOU_THRESH).float()
            all_recall.append(gt_recalled.mean().item())

            # Spearman: match each GT box to best pred (greedy IoU)
            pred_idx, gt_idx = hungarian_match(pb.detach(), gt_boxes)
            all_pred_scores.extend(ps[pred_idx].cpu().tolist())
            all_gt_weights.extend(gt_weights[gt_idx].cpu().tolist())

    val_box   = total_box   / n_batches
    val_score = total_score / n_batches

    recall = float(np.mean(all_recall)) if all_recall else 0.0

    spearman = 0.0
    if len(all_pred_scores) > 5:
        spearman = spearmanr(all_pred_scores, all_gt_weights).statistic

    return {
        "val_loss":       val_box + val_score,
        "val_box_loss":   val_box,
        "val_score_loss": val_score,
        f"recall@{RECALL_K}@{IOU_THRESH}": recall,
        "score_spearman": spearman,
    }


def visualise_heatmaps(model, ds, class_to_idx, n=N_HEATMAPS):
    import matplotlib.pyplot as plt
    from src.viz import prepare_image

    sigma_scale    = 0.35
    truncate_sigma = 4.0

    model.eval()
    os.makedirs("output/heatmaps", exist_ok=True)

    xs_full = torch.arange(IMAGE_SIZE)
    ys_full = torch.arange(IMAGE_SIZE)

    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

    with torch.no_grad():
        for i in range(n):
            sample    = ds[i]
            image     = sample["image"].unsqueeze(0).to(DEVICE)
            class_idx = torch.tensor(
                [class_to_idx.get(sample['class'], 0)], dtype=torch.long, device=DEVICE
            )

            pred_boxes, pred_scores = forward(model, image, class_idx)
            pred_boxes  = pred_boxes[0]   # (100, 4) normalised cxcywh
            pred_scores = pred_scores[0]  # (100,)

            heatmap = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
            for (cx, cy, w, h), score in zip(pred_boxes, pred_scores):
                # convert normalised → pixel space
                cx = (cx * IMAGE_SIZE).item()
                cy = (cy * IMAGE_SIZE).item()
                w  = (w  * IMAGE_SIZE).item()
                h  = (h  * IMAGE_SIZE).item()

                if w <= 0 or h <= 0:
                    continue

                sigma_x  = sigma_scale * w
                sigma_y  = sigma_scale * h
                radius_x = int(np.ceil(truncate_sigma * sigma_x))
                radius_y = int(np.ceil(truncate_sigma * sigma_y))

                x0 = max(0, int(np.floor(cx - radius_x)))
                x1 = min(IMAGE_SIZE, int(np.ceil(cx + radius_x + 1)))
                y0 = max(0, int(np.floor(cy - radius_y)))
                y1 = min(IMAGE_SIZE, int(np.ceil(cy + radius_y + 1)))

                xs = xs_full[x0:x1].view(1, -1)
                ys = ys_full[y0:y1].view(-1, 1)

                g = torch.exp(
                    -0.5 * (
                        ((xs - cx) / sigma_x) ** 2 +
                        ((ys - cy) / sigma_y) ** 2
                    )
                )
                heatmap[y0:y1, x0:x1] += score.item() * g

            heatmap = heatmap / (heatmap.max() + 1e-8)

            bg = prepare_image(sample["image"])

            # Left column: background only
            axes[i, 0].imshow(bg)
            axes[i, 0].set_title(f"class='{sample['class']}'", fontsize=8)
            axes[i, 0].axis("off")

            # Right column: background with heatmap overlaid
            axes[i, 1].imshow(bg)
            axes[i, 1].imshow(heatmap.numpy(), cmap="jet", alpha=0.5)
            axes[i, 1].set_title("predicted heatmap", fontsize=8)
            axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = "output/heatmaps/all_heatmaps.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def main():
    os.makedirs("output", exist_ok=True)

    wandb.init(
        project="adlcv-detr-placement",
        config={
            "split":       SPLIT,
            "image_size":  IMAGE_SIZE,
            "batch_size":  BATCH_SIZE,
            "max_gt":      MAX_GT,
            "lr":          LR,
            "epochs":      EPOCHS,
            "recall_k":    RECALL_K,
            "iou_thresh":  IOU_THRESH,
            "backbone":    "resnet-50",
            "freeze":      "backbone+encoder",
        },
    )

    print("Loading datasets...")
    train_ds = HiddenObjectsImageClassLevelFast(split="train", image_size=IMAGE_SIZE)
    val_ds   = HiddenObjectsImageClassLevelFast(split="test",  image_size=IMAGE_SIZE)

    # Build class vocabulary from training set
    unique_classes = sorted(set(train_ds.class_arr.tolist()))
    class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
    num_classes    = len(unique_classes)
    print(f"  Found {num_classes} object classes")

    collate_fn = make_collate_fn(class_to_idx)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    print("Building model...")
    model = build_model(num_classes).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)

    global_step = 0

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, class_indices, targets in pbar:
            images        = images.to(DEVICE)
            class_indices = class_indices.to(DEVICE)

            pred_boxes, pred_scores = forward(model, images, class_indices)

            box_loss, score_loss = compute_loss(pred_boxes, pred_scores, targets)
            loss = box_loss + score_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            global_step += 1

            wandb.log({
                "train/loss":       loss.item(),
                "train/box_loss":   box_loss.item(),
                "train/score_loss": score_loss.item(),
            }, step=global_step)

            pbar.set_postfix(
                box=f"{box_loss.item():.4f}",
                score=f"{score_loss.item():.4f}",
                loss=f"{loss.item():.4f}",
            )

        train_avg = total_loss / len(train_dl)

        # --- Validate ---
        metrics = evaluate(model, val_dl, desc=f"Val E{epoch+1}")

        recall_key = f"recall@{RECALL_K}@{IOU_THRESH}"
        print(
            f"Epoch {epoch+1}  "
            f"train={train_avg:.4f}  "
            f"val={metrics['val_loss']:.4f}  "
            f"(box={metrics['val_box_loss']:.4f} score={metrics['val_score_loss']:.4f})  "
            f"{recall_key}={metrics[recall_key]:.3f}  "
            f"spearman={metrics['score_spearman']:.3f}"
        )

        wandb.log({
            "epoch":                    epoch + 1,
            "train/epoch_loss":         train_avg,
            "val/loss":                 metrics["val_loss"],
            "val/box_loss":             metrics["val_box_loss"],
            "val/score_loss":           metrics["val_score_loss"],
            f"val/{recall_key}":        metrics[recall_key],
            "val/score_spearman":       metrics["score_spearman"],
        }, step=global_step)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

    print("Generating heatmaps...")
    visualise_heatmaps(model, train_ds, class_to_idx)

    wandb.log({"heatmaps": wandb.Image("output/heatmaps/all_heatmaps.png")})
    wandb.finish()


if __name__ == "__main__":
    main()
