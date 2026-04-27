import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, feat_dim, cond_dim):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, feat_dim * 2),
            nn.SiLU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )

    def forward(self, x, cond):
        # x: [B, C, H, W]
        # cond: [B, D]
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.film = FiLM(out_ch, cond_dim) if cond_dim else None

    def forward(self, x, cond=None):
        x = self.conv(x)
        if self.film is not None:
            x = self.film(x, cond)
        return x


class CrossAttention2D(nn.Module):
    """
    Lightweight query-conditioned spatial attention.
    Dense image tokens attend to one class token.
    """
    def __init__(self, feat_dim, cond_dim, heads=4):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, feat_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x, cond):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose(1, 2)       # [B, HW, C]
        query = self.cond_proj(cond).unsqueeze(1)   # [B, 1, C]

        # image tokens attend to class token
        out, _ = self.attn(
            query=tokens,
            key=query,
            value=query,
        )

        tokens = self.norm(tokens + out)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class FrozenDINOFeatureAdapter(nn.Module):
    """
    Pseudocode wrapper.

    Use a small frozen DINO:
      - DINOv2 ViT-S/14 or ViT-B/14
      - frozen
      - output patch feature map, e.g. [B, C, 36, 36] for 512 input
    """
    def __init__(self, dino_model, dino_dim, out_dim=256):
        super().__init__()
        self.dino = dino_model
        self.proj = nn.Conv2d(dino_dim, out_dim, 1)

        for p in self.dino.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_dino(self, rgb):
        """
        Return dense patch features.

        Expected:
            rgb: [B, 3, 512, 512]
            return: [B, C_dino, H/14, W/14]
        """
        patch_tokens = self.dino.forward_features(rgb)["x_norm_patchtokens"]
        B, N, C = patch_tokens.shape

        h = w = int(N ** 0.5)
        feat = patch_tokens.transpose(1, 2).reshape(B, C, h, w)
        return feat

    def forward(self, rgb):
        with torch.no_grad():
            feat = self.extract_dino(rgb)

        # projection is trainable and cheap
        return self.proj(feat)


class FrozenDepthAdapter(nn.Module):
    """
    Frozen Depth Anything V2 + small trainable geometry encoder.
    """
    def __init__(self, depth_model, out_dim=128):
        super().__init__()
        self.depth_model = depth_model

        for p in self.depth_model.parameters():
            p.requires_grad = False

        self.encoder = nn.Sequential(
            ConvBlock(5, 32),
            nn.AvgPool2d(2),      # 256
            ConvBlock(32, 64),
            nn.AvgPool2d(2),      # 128
            ConvBlock(64, 96),
            nn.AvgPool2d(2),      # 64
            ConvBlock(96, out_dim),
        )

    @torch.no_grad()
    def predict_depth(self, rgb):
        """
        Return normalized relative depth:
            [B, 1, 512, 512]
        """
        depth = self.depth_model(rgb)

        if depth.ndim == 3:
            depth = depth[:, None]

        d_min = depth.amin(dim=(2, 3), keepdim=True)
        d_max = depth.amax(dim=(2, 3), keepdim=True)
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        return depth

    def forward(self, rgb):
        with torch.no_grad():
            depth = self.predict_depth(rgb)

        dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        B, _, H, W = depth.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=depth.device),
            torch.linspace(-1, 1, W, device=depth.device),
            indexing="ij",
        )

        coord = torch.stack([xx, yy], dim=0)[None].repeat(B, 1, 1, 1)

        geom = torch.cat([depth, dx, dy, coord], dim=1)  # [B, 5, H, W]
        return self.encoder(geom), depth


class PlacementDecoder(nn.Module):
    def __init__(self, in_ch, cond_dim, base=128):
        super().__init__()

        self.block0 = ConvBlock(in_ch, base * 2, cond_dim)
        self.attn0 = CrossAttention2D(base * 2, cond_dim)

        self.up1 = ConvBlock(base * 2, base, cond_dim)
        self.attn1 = CrossAttention2D(base, cond_dim)

        self.up2 = ConvBlock(base, base // 2, cond_dim)
        self.up3 = ConvBlock(base // 2, base // 4, cond_dim)

        self.out = nn.Sequential(
            nn.Conv2d(base // 4, base // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base // 4, 1, 1),
        )

    def forward(self, x, cond):
        # x is around 36×36 or 64×64 depending on chosen feature resolution

        x = self.block0(x, cond)
        x = self.attn0(x, cond)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x, cond)
        x = self.attn1(x, cond)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x, cond)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x, cond)

        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

        logits = self.out(x)
        return logits


class PlaceNet(nn.Module):
    """
    Main model.

    Inputs:
        rgb:       [B, 3, 512, 512]
        class_id:  [B]

    Output:
        logits:    [B, 1, 512, 512]
        heatmap:   sigmoid(logits)
    """
    def __init__(
        self,
        dino_model,
        depth_model,
        num_classes=50,
        dino_dim=384,          # ViT-S/14; use 768 for ViT-B/14
        class_dim=128,
        hidden=256,
    ):
        super().__init__()

        self.class_emb = nn.Embedding(num_classes, class_dim)

        self.class_mlp = nn.Sequential(
            nn.Linear(class_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.rgb_encoder = FrozenDINOFeatureAdapter(
            dino_model=dino_model,
            dino_dim=dino_dim,
            out_dim=hidden,
        )

        self.depth_encoder = FrozenDepthAdapter(
            depth_model=depth_model,
            out_dim=128,
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + 128 + 1, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )

        self.fuse_film = FiLM(hidden, hidden)

        self.decoder = PlacementDecoder(
            in_ch=hidden,
            cond_dim=hidden,
            base=128,
        )

    def forward(self, rgb, class_id):
        cond = self.class_mlp(self.class_emb(class_id))

        rgb_feat = self.rgb_encoder(rgb)          # [B, 256, ~36, ~36]
        depth_feat, depth = self.depth_encoder(rgb)

        target_hw = rgb_feat.shape[-2:]

        depth_feat = F.interpolate(
            depth_feat,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

        depth_low = F.interpolate(
            depth,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([rgb_feat, depth_feat, depth_low], dim=1)
        x = self.fuse(x)
        x = self.fuse_film(x, cond)

        logits = self.decoder(x, cond)
        heatmap = torch.sigmoid(logits)

        return {
            "logits": logits,
            "heatmap": heatmap,
        }