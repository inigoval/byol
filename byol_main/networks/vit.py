import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads

        # If multi-headed attention, project dimensions back to initial dimension at the end,
        # otherwise just pass through
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        # Large dimensions will cause the softmax to have extremely small gradients,
        # so the dot product scales with the dimension
        self.scale = dim_head**-0.5

        # Softmax to calculate weights for each value
        self.attend = nn.Softmax(dim=-1)

        # Attention dropout drops out after the softmax
        self.dropout = nn.Dropout(dropout)

        # Convert linear projection of patches to queries keys and values
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Project to original dimension if using multi-headed attention
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        # Split into q, k and v chunks
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # Split q, k, v across multiple heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Dot product of q and k with scaling
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply softmax and attention dropout
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Multiply attention weights with values
        out = torch.matmul(attn, v)

        # Recombine all heads into single vector with dim=inner_dim
        out = rearrange(out, "b h n d -> b n (h d)")

        # Project out to correct number  of dimensions and return
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            # Apply attention and add input
            x = attn(x) + x
            # Run through feedforward network add input
            x = ff(x) + x
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height = image_width = image_size
        patch_height = patch_width = patch_size

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
