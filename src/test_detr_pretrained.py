import sys, os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
from tqdm import tqdm

from src.hidden_objects_dataset import HiddenObjectsImageClassLevelFast

# --- Config ---
SPLIT = "test"      # use test split locally (smaller)
IMAGE_SIZE = 512
BATCH_SIZE = 4
MAX_GT = 50         # max positive boxes per sample to match against
LR = 1e-4
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "output/detr_placement.pt"

# Model: load, freeze, replace head
def build_model():
    detr = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50"
    )

    #Freeze backbone and encoder
    for param in detr.model.backbone.parameters():
        param.requires_grad = False
    for param in detr.model.encoder.parameters():
        param.requires_grad = False

    # Replace 92-class head with a single scalar score head 
    hidden_dim = detr.config.d_model    # 256
    detr.score_head = nn.Linear(hidden_dim, 1)

    return detr

# Collate function (inline, no seperate file)
def collate_fn(batch):
    images = torch.stack([s["image"] for s in batch])   # (B, 3, H, W)

    targets = []
    for s in batch:
        pos_mask = s['labels'].bool()
        boxes = s['boxes'][pos_mask]                    # pixel xywh
        confidences = s['confidences'][pos_mask]
        rewards = s['image_reward_scores'][pos_mask]

        # weight = confidence * normalised_reward
        r_min, r_max = rewards.min(), rewards.max()
        norm_rewards = torch.ones_like(rewards) if r_min == r_max else (rewards - r_min) / (r_max - r_min)
        weights = confidences * norm_rewards            # (K,)

        # keep top MAX_GT by weight
        if len(weights) > MAX_GT:
            _, keep = weights.topk(MAX_GT)
            boxes, weights = boxes[keep], weights[keep]

        # convert to normalised cxcywh for DETR
        x, y, w, h = boxes.unbind(-1)
        cx = (x + 0.5*w) / IMAGE_SIZE
        cy = (y + 0.5*h) / IMAGE_SIZE
        wn = w / IMAGE_SIZE
        hn = h / IMAGE_SIZE
        boxes_norm = torch.stack([cx, cy, wn, hn], dim=-1).clamp(0,1)

        targets.append({"boxes": boxes_norm, "weights": weights})

    return images, targets

# Hungarian matching
def hungarian_match(pred_boxes, gt_boxes):
    """
    pred_boxes: (100, 4)    normalized cxcywh
    gt_boxes: (K,4)         normalized cxcywh
    returns: (pred_indices, gt_indices)
    """
    # Cost matrix = L1 distance between every pred and every Gt box
    cost = torch.cdist(pred_boxes, gt_boxes, p=1).cpu().numpy()    # (100, K)
    pred_idx, gt_idx = linear_sum_assignment(cost)
    return pred_idx, gt_idx

# Loss
def compute_loss(pred_boxes, pred_scores, targets):
    """
    pred_boxes:     (B, 100, 4)
    pred_scores:    (B, 100)
    targets:        list of B dicts with keys 'boxes', 'weights'
    """
    total_box_loss = 0.0
    total_score_loss = 0.0

    for b in range(len(targets)):
        gt_boxes = targets[b]['boxes'].to(pred_boxes.device)
        gt_weights = targets[b]['weights'].to(pred_boxes.device)

        pb = pred_boxes[b]      # (100, 4)
        ps = pred_scores[b]     # (100,)

        score_targets = torch.zeros(100, device=pb.device)

        if len(gt_boxes) > 0: 
            pred_idx, gt_idx = hungarian_match(pb.detach(), gt_boxes)

            # box loss on matched pairs
            total_box_loss += nn.functional.l1_loss(
                pb[pred_idx], gt_boxes[gt_idx]
            )

            # score targets for matched queries
            score_targets[pred_idx] = gt_weights[gt_idx].float()

        # score loss on all 100 queries
        total_score_loss += nn.functional.mse_loss(ps, score_targets)

    n = len(targets)
    return total_box_loss / n, total_score_loss / n

def visualise_heatmaps(model, ds, n=4):
    import matplotlib.pyplot as plt
    from src.viz import prepare_image
    import os

    model.eval()
    os.makedirs("output/heatmaps", exist_ok=True)

    with torch.no_grad():
        for i in range(n):
            sample = ds[i]
            image  = sample["image"].unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

            outputs  = model.model(pixel_values=image)
            seq_out  = outputs.last_hidden_state
            pred_boxes  = model.bbox_predictor(seq_out).sigmoid()[0]   # (100, 4) normalised cxcywh
            pred_scores = model.score_head(seq_out).squeeze(-1).sigmoid()[0]  # (100,)

            # Build heatmap — paint each box weighted by its score
            heatmap = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
            for (cx, cy, w, h), score in zip(pred_boxes, pred_scores):
                cx, cy, w, h = (cx*IMAGE_SIZE).item(), (cy*IMAGE_SIZE).item(), \
                               (w*IMAGE_SIZE).item(),  (h*IMAGE_SIZE).item()
                x0 = max(0, int(cx - w/2))
                y0 = max(0, int(cy - h/2))
                x1 = min(IMAGE_SIZE, int(cx + w/2))
                y1 = min(IMAGE_SIZE, int(cy + h/2))
                heatmap[y0:y1, x0:x1] += score.item()

            heatmap = heatmap / (heatmap.max() + 1e-8)

            bg = prepare_image(sample["image"])
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"class='{sample['class']}'")
            axes[0].imshow(bg)
            axes[0].set_title("Background")
            axes[0].axis("off")
            axes[1].imshow(heatmap.numpy(), cmap="jet")
            axes[1].set_title("Predicted placement heatmap")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(f"output/heatmaps/sample_{i}_{sample['class']}.png", dpi=120)
            plt.close()
            print(f"  Saved heatmap {i} → output/heatmaps/sample_{i}_{sample['class']}.png")


def main():
    os.makedirs("output", exist_ok=True)

    print("Loading dataset...")
    ds = HiddenObjectsImageClassLevelFast(split=SPLIT, image_size=IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate_fn, num_workers=0)

    print("Building model...")
    model = build_model().to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, (images, targets) in enumerate(pbar):
            images = images.to(DEVICE)

            # forward — reuse DETR's internal forward up to decoder output
            outputs = model.model(pixel_values=images)
            seq_out = outputs.last_hidden_state          # (B, 100, 256)

            pred_boxes   = model.bbox_predictor(seq_out).sigmoid()  # (B, 100, 4)
            pred_scores  = model.score_head(seq_out).squeeze(-1).sigmoid()  # (B, 100)

            box_loss, score_loss = compute_loss(pred_boxes, pred_scores, targets)
            loss = box_loss + score_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(box=f"{box_loss.item():.4f}", score=f"{score_loss.item():.4f}", loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1} avg loss: {total_loss / len(dl):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

    print("Generating heatmaps...")
    visualise_heatmaps(model, ds)

if __name__ == "__main__":
    main()

