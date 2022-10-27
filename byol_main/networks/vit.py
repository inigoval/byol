import torch
from torch import nn
import logging
import torch.nn.functional as F

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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, attention_layer=Attention):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, attention_layer(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
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
        # num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        # If image isn't square, get dimensions from tuple
        # otherwise for square image just copy single dimension for square image
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # Check that image can be split up into patches of given size
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        # Calculate total number of patches
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        # Calculate total number of values in each patch
        patch_dim = channels * patch_height * patch_width

        # Turn (c x h x w) images into (n_patchs x patch_dim) flattened patches and
        # project to latent dimension
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n1 p1) (n2 p2) -> b (n1 n2) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        # Positional embedding is learned in ViT. Note that timm implementation multiplies normal distribution initialization by 0.02
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Initialize cls token which is used as representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Initialize transformer with given parameters
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # For finetuning
        self.finetuning_layers = self.transformer.layers

        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        # Representation dimension (for finetuning etc)
        self.dim = dim

    def forward(self, x):
        # Turn image into flattened patches
        x = self.to_patch_embedding(x)

        # Get batch size and number of patches
        b, n, _ = x.shape

        # Expand cls token to correct size
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)

        # Combine cls tokens with patches
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding and dropout
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        # Run embedding through transformer
        x = self.transformer(x)

        # Get new class token
        x = x[:, 0]

        #  Return class token after passing through MLP
        x = self.to_latent(x)
        # return self.mlp_head(x)

        return x


#############################
## ViT for small data-sets ##
#############################


# class ViT(nn.Module):
#     def __init__(
#         self,
#         *,
#         image_size,
#         patch_size,
#         # num_classes,
#         dim,
#         depth,
#         heads,
#         mlp_dim,
#         channels=3,
#         dim_head=64,
#         dropout=0.0,
#         emb_dropout=0.0,
#         arch="vit"  # ['vit', 'vit-small']
#     ):
#         super().__init__()
#         # If image isn't square, get dimensions from tuple
#         # otherwise for square image just copy single dimension for square image
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)

#         # Check that image can be split up into patches of given size
#         assert (
#             image_height % patch_height == 0 and image_width % patch_width == 0
#         ), "Image dimensions must be divisible by the patch size."

#         # Calculate total number of patches
#         num_patches = (image_height // patch_height) * (image_width // patch_width)

#         # Calculate total number of values in each patch
#         patch_dim = channels * patch_height * patch_width

#         # Turn (c x h x w) images into (n_patchs x patch_dim) flattened patches
#         # self.to_patch = nn.Sequential(
#         # Rearrange("b c (n1 p1) (n2 p2) -> b (n1 n2) (p1 p2 c)", p1=patch_height, p2=patch_width),
#         # )

#         # project to latent dimension
#         self.to_patch = Rearrange(
#             "b c (n1 p1) (n2 p2) -> b (n1 n2) (p1 p2 c)", p1=patch_height, p2=patch_width
#         )

#         if arch == "vit":
#             self.to_patch_emb = nn.Sequential(
#                 Rearrange("b c (n1 p1) (n2 p2) -> b (n1 n2) (p1 p2 c)", p1=patch_height, p2=patch_width),
#                 nn.Linear(patch_dim, dim),
#             )
#         elif arch == "vit-small":
#             self.to_patch_emb = SPT(dim=dim, patch_size=patch_size, channels=channels)

#         # Positional embedding is learned in ViT.
#         # Note that timm implementation multiplies normal distribution initialization by 0.02
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

#         # Initialize cls token which is used as representation
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         # Initialize transformer with given parameters
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.to_latent = nn.Identity()

#         # self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

#     def forward(self, x):
#         # Turn image into flattened patches
#         x = self.to_patch_emb(x)

#         # Get batch size and number of patches
#         b, n, _ = x.shape

#         # Expand cls token to correct size
#         cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)

#         # Combine cls tokens with patches
#         x = torch.cat((cls_tokens, x), dim=1)

#         # Add positional embedding and dropout
#         x += self.pos_embedding[:, : (n + 1)]
#         x = self.dropout(x)

#         # Run embedding through transformer
#         x = self.transformer(x)

#         # Get new class token
#         x = x[:, 0]

#         #  Return class token after passing through MLP
#         x = self.to_latent(x)
#         # return self.mlp_head(x)

#         return x


class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels=3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)


class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head**-0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
