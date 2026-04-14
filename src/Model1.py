# using openAI https://huggingface.co/openai/clip-vit-base-
# And https://huggingface.co/facebook/dinov2-base
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from src._datasets import HiddenObjectsHeatmap, heatmap_collate
from transformers import AutoImageProcessor, AutoModel
import src.viz as viz
import matplotlib.pyplot as plt
from src.globals import HF_CACHE_DIR, PLACES365_ROOT, PLACES365_TRIMMED_ROOT, DATA_ROOT
import random
import time

CACHE_DIR = "/work3/s215225/hf_cache"
os.environ["HF_HOME"] = "/work3/s215225/hf_cache"

# TODO wtite our own pre processor that does the same thing but uses our data 
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=CACHE_DIR)
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir=CACHE_DIR)

def preprocess_texts(texts, device="cuda"):
    # clip_processor: BPE-tokenizes text into subword token ids (vocab=49408, max_len=77),
    # wraps with SOS/EOS tokens, pads the batch to equal length, returns input_ids (B, seq_len)
    if isinstance(texts, str):
        texts = [texts]
    tokens = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    return tokens["input_ids"].to(device)

def preprocess_images(images, device="cuda"):
    return dino_processor(images=images, return_tensors="pt")["pixel_values"].to(device)

class TextEncoder(torch.nn.Module):
    """
    CLIP text encoder. Accepts raw strings, tokenizes internally.

    Layers:
        token_embedding   : Embedding(49408, 512)   — maps token ids to vectors
        position_embedding: Embedding(77, 512)       — one pos per token (max 77)
        layers            : 12x transformer block
                              .layer_norm1  LayerNorm(512)
                              .self_attn    q_proj / k_proj / v_proj / out_proj  Linear(512, 512)
                              .layer_norm2  LayerNorm(512)
                              .mlp.fc1      Linear(512, 2048)
                              .mlp.fc2      Linear(2048, 512)
        final_layer_norm  : LayerNorm(512)
        projection        : Linear(512, 512, bias=False)
    """
    def __init__(self, freeze=True, device="cuda"):
        super().__init__()
        self.device = device
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=CACHE_DIR)
        txt  = clip.text_model

        self.token_embedding    = txt.embeddings.token_embedding    # Embedding(49408, 512)
        self.position_embedding = txt.embeddings.position_embedding # Embedding(77, 512)
        self.position_ids       = txt.embeddings.position_ids
        self.layers             = txt.encoder.layers                # 12x CLIPEncoderLayer
        self.final_layer_norm   = txt.final_layer_norm              # LayerNorm(512)
        self.projection         = clip.text_projection              # Linear(512, 512)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        self.to(device)

    def forward(self, input_ids):
        # input_ids: (B, seq_len) — from preprocess_texts()
        B, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)                          # (B, seq_len, 512)
        x = x + self.position_embedding(self.position_ids[:, :seq_len].to(self.device))  # (B, seq_len, 512)
        for layer in self.layers:
            x = layer(x, attention_mask=None, causal_attention_mask=None)[0]
        x = self.final_layer_norm(x)

        # take the EOS token position (highest token id = EOS in CLIP)
        eos_idx = input_ids.argmax(dim=-1)
        x = x[torch.arange(B), eos_idx]   # (B, 512)
        return self.projection(x)          # (B, 512)

class DinoVisionEncoder(torch.nn.Module):
    """
    DINOv2 ViT-B/14 vision encoder (facebook/dinov2-base).

    Input : (B, 3, 224, 224)
    Patch size  : 14x14 pixels  →  224/14 = 16, so a 16x16 grid = 256 patches
    Each patch  : projected to 768-dim vector

    Layers:
        embeddings.patch_embeddings.projection : Conv2d(3, 768, kernel=14, stride=14)
        embeddings.cls_token                   : Parameter(1, 1, 768)
        embeddings.position_embeddings         : Parameter(1, 257, 768)  (256 patches + 1 CLS)
        encoder.layer                          : 12x transformer block
                              .norm1                               LayerNorm(768)
                              .attention.attention.query/key/value Linear(768, 768)
                              .attention.output.dense              Linear(768, 768)
                              .norm2                               LayerNorm(768)
                              .intermediate.dense                  Linear(768, 3072)
                              .output.dense                        Linear(3072, 768)
        layernorm                              : LayerNorm(768)

    Output:
        cls_token    : (B, 768)        — global image vector
        patch_tokens : (B, 256, 768)   — 16x16 spatial grid, each cell 768-dim
    """
    def __init__(self, freeze=True, device="cuda"):
        super().__init__()
        self.device = device

        dino = AutoModel.from_pretrained('facebook/dinov2-base', cache_dir=CACHE_DIR)

        self.embeddings  = dino.embeddings   # patch + cls + position embeddings
        self.encoder     = dino.encoder      # 12x transformer layers
        self.layernorm   = dino.layernorm    # final LayerNorm(768)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        self.to(device)

    def forward(self, pixel_values):
        # pixel_values: (B, 3, H, W) — from preprocess_images()
        x = self.embeddings(pixel_values)               # (B, num_patches+1, 768)
        x = self.encoder(x).last_hidden_state           # (B, num_patches+1, 768)
        x = self.layernorm(x)
        cls_token    = x[:, 0, :]   # (B, 768)  — global image representation
        patch_tokens = x[:, 1:, :]  # (B, num_patches, 768) — spatial grid
        return cls_token, patch_tokens

class Decoder(torch.nn.Module):
    """
    TransformerDecoderLayer + CNN decoder.

    Step 1 — Project both inputs to common dim (256):
        patch_tokens  (B, 256, 768) → Linear(768, 256) → (B, 256, 256)
        text_features (B, 512)      → Linear(512, 256) → (B,   1, 256)

    Step 2 — TransformerDecoderLayer (1 layer, d_model=256):
        each patch attends to the text → spatially varying output (B, 256, 256)

    Step 3 — reshape to spatial map: (B, 256, 16, 16)
        C=256 patches, each patch's 256 features unfold to a 16x16 grid

    Step 4 — 4x CNN upsample to (B, 1, 512, 512):
        Conv(256, 128) + ReLU + Upsample x2  →  (B, 128,  32,  32)
        Conv(128,  64) + ReLU + Upsample x4  →  (B,  64, 128, 128)
        Conv( 64,  32) + ReLU + Upsample x2  →  (B,  32, 256, 256)
        Conv( 32,   1)        + Upsample x2  →  (B,   1, 512, 512)
        sigmoid                               →  (B, 512, 512) in [0, 1]
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.patch_proj = torch.nn.Linear(768, 256)
        self.text_proj  = torch.nn.Linear(512, 256)
        self.transformer_layer = torch.nn.TransformerDecoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024,
            batch_first=True, dropout=0.0,
        )

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            torch.nn.Conv2d(128,  64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            
            torch.nn.Conv2d( 64,  32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            torch.nn.Conv2d( 32,   1, kernel_size=3, padding=1),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.to(device)

    def forward(self, patch_tokens, text_features):
        tgt = self.patch_proj(patch_tokens)               # (B, 256, 256)
        mem = self.text_proj(text_features).unsqueeze(1)  # (B,   1, 256)

        x = self.transformer_layer(tgt=tgt, memory=mem)   # (B, 256, 256)
        
        x = x.view(patch_tokens.shape[0], 256, 16, 16)                        # (B, 256, 16, 16)
        
        heatmap = self.cnn(x).squeeze(1)                   # (B, 512, 512)
        return torch.sigmoid(heatmap)

class HeatmapModel(torch.nn.Module):
    def __init__(self, device="cuda", freeze_text_encoder=False, freeze_vision_encoder=False):
        super().__init__()
        self.device = device
        self.freeze_text_encoder   = freeze_text_encoder
        self.freeze_vision_encoder = freeze_vision_encoder
        self.text_encoder   = TextEncoder(freeze=freeze_text_encoder, device=device)
        self.vision_encoder = DinoVisionEncoder(freeze=freeze_vision_encoder, device=device)
        self.decoder        = Decoder(device=device)

    def forward(self, images, texts):
        if self.freeze_text_encoder:
            with torch.no_grad():
                text_features = self.text_encoder(preprocess_texts(texts, self.device))
        else:
            text_features = self.text_encoder(preprocess_texts(texts, self.device))

        if self.freeze_vision_encoder:
            with torch.no_grad():
                _, patch_tokens = self.vision_encoder(preprocess_images(images, self.device))
        else:
            _, patch_tokens = self.vision_encoder(preprocess_images(images, self.device))

        return self.decoder(patch_tokens, text_features)         # (B, 512, 512)



def loss_fn(pred_heatmaps, target_heatmaps):
    # TODO a better loss function, one that works with probability distributions
    return  F.binary_cross_entropy(pred_heatmaps, target_heatmaps)

def get_data(batch_size=32):
    train_dataset = HiddenObjectsHeatmap(split="train")
    test_dataset = HiddenObjectsHeatmap(split="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=heatmap_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=heatmap_collate)
    return train_loader, test_loader

def train(model, train_loader, optimizer, num_epochs=10, test_loader=None, save_path=None):
    best_test_loss = float("inf")
    print(f"Training for {num_epochs} epochs | batches/epoch: {len(train_loader)} | save: {save_path}", flush=True)
    model.train()
    for epoch in range(num_epochs):
        t0 = time.time()
        total_train_loss = 0.0
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(model.device)
            heatmaps = batch["heatmap"].to(model.device)
            texts = batch["class"]
            pred_heatmaps = model(images, texts)
            loss = loss_fn(pred_heatmaps, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        elapsed = time.time() - t0

        if test_loader is not None:
            avg_test_loss = test_model(model, test_loader)
            saved = ""
            if save_path is not None and avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), save_path)
                saved = " [saved]"
            print(f"Epoch {epoch+1}/{num_epochs} | train: {avg_train_loss:.4f} | test: {avg_test_loss:.4f}{saved} | {elapsed:.0f}s", flush=True)
            model.train()
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | train: {avg_train_loss:.4f} | {elapsed:.0f}s", flush=True)

def test_traning(model, train_loader):
    batch = next(iter(train_loader))
    print("Data loaded. Starting training...")
    images = batch["image"].to(model.device)
    heatmaps = batch["heatmap"].to(model.device)
    texts = batch["class"]
    print(texts)
    print(images.shape, heatmaps.shape)
    print("Running forward pass...")
    pred_heatmaps = model(images, texts)
    loss = loss_fn(pred_heatmaps, heatmaps)
    print(f"Initial loss: {loss.item():.4f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Test training step done.")

def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0
        for batch in test_loader:
            images = batch["image"].to(model.device)
            heatmaps = batch["heatmap"].to(model.device)
            texts = batch["class"]
            pred_heatmaps = model(images, texts)
            loss = loss_fn(pred_heatmaps, heatmaps)
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    print(f"Average Test Loss: {avg_loss:.4f}")
    return avg_loss

def visualize_predictions(model, test_loader, num_samples=4, seed=42, object_name=None):
    rng = random.Random(seed)
    dataset = test_loader.dataset
    if object_name is not None:
        indices = [i for i in range(len(dataset)) if dataset[i]["class"] == object_name]
    else:
        indices = list(range(len(dataset)))

    chosen = rng.sample(indices, num_samples)
    samples = [dataset[i] for i in chosen]

    images  = torch.stack([s["image"]   for s in samples]).to(model.device)
    heatmaps = torch.stack([s["heatmap"] for s in samples]).to(model.device)
    texts   = [s["class"] for s in samples]

    model.eval()
    with torch.no_grad():
        pred_heatmaps = model(images, texts)

    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    for i in range(num_samples):
        img          = viz.prepare_image(images[i])
        pred_overlay = viz.overlay_heatmap_on_image(img, pred_heatmaps[i].cpu().numpy())
        gt_overlay   = viz.overlay_heatmap_on_image(img, heatmaps[i].cpu().numpy())

        axes[0, i].imshow(pred_overlay)
        axes[0, i].set_title(f"pred: {texts[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(gt_overlay)
        axes[1, i].set_title("gt")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--freeze-text-encoder", action="store_true")
    parser.add_argument("--freeze-vision-encoder", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--save-model-path", type=str, default=None)
    parser.add_argument("--load-model-path", type=str, default=None)


    parser.add_argument("--mode", type=str, choices=["train", "test", "visualize"], default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test-train", action="store_true")
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    freeze_text_encoder = args.freeze_text_encoder
    freeze_vision_encoder = args.freeze_vision_encoder
    device = args.device

    lr = args.lr

    test_train = args.test_train

    mode = args.mode
    
    if args.load_model_path:
        model = HeatmapModel(device=device, freeze_text_encoder=freeze_text_encoder, freeze_vision_encoder=freeze_vision_encoder)
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        print(f"Model loaded from {args.load_model_path}")
    else:
        model = HeatmapModel(device=device, freeze_text_encoder=freeze_text_encoder, freeze_vision_encoder=freeze_vision_encoder)
    train_loader, test_loader = get_data(batch_size=batch_size)
    print("Data loaded")
    
    if test_train:
        test_traning(model, train_loader)
    
    if mode == "train":
        print("Starting training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        save_path = DATA_ROOT / (args.save_model_path or "best_model.pth")
        train(model, train_loader, optimizer, num_epochs=num_epochs, test_loader=test_loader, save_path=save_path)
    elif mode == "test":
            test_model(model, test_loader)
    elif mode == "visualize":
        visualize_predictions(model, test_loader)


    # # From https://huggingface.co/openai/clip-vit-base-patch16
    # cache_dir = "/work3/s215225/hf_cache"
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=cache_dir)
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=cache_dir)
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    # outputs = model(**inputs)
    # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    # print(probs)
    # ####
    # image = Image.open(requests.get(url, stream=True).raw)
    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    # model = AutoModel.from_pretrained('facebook/dinov2-base')
    # inputs = processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state
    # ##########




