import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from byol_main.networks.vit import Transformer, ViT
from byol_main.evaluation import Lightning_Eval
from utilities import _optimizer


class MAE(Lightning_Eval):
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters()  # save hyperparameters for easy inference
        self.config = config

        self.masking_ratio = config["model"]["masking_ratio"]

        # Initialize encoder and extract dimensions
        self.encoder = ViT(image_size=self.config["data"]["input_height"], **self.config["model"]["vit"])
        self.to_patch, self.patch_to_emb = self.encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        num_patches, enc_dim = self.encoder.pos_embedding.shape[-2:]

        # Projection to match encoder/decoder dimensions
        dec_dim = self.config["model"]["decoder"]["dim"]
        self.enc2dec = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()

        # Initialize decoder
        self.config["model"]["decoder"]["mlp_dim"] = self.config["model"]["decoder"]["dim"] * 4
        self.decoder = Transformer(**self.config["model"]["decoder"]).to(self.device)

        # Masking tokens for decoder.
        self.mask_token = nn.Parameter(torch.randn(dec_dim))

        # Fixed embeddings for decoder
        self.decoder_pos_emb = nn.Embedding(num_patches, dec_dim)
        self.to_pixels = nn.Linear(dec_dim, pixel_values_per_patch)

    def forward(self, x):
        return self.encoder(x)  # dimension (batch, features), features from config e.g. 512

    def training_step(self, batch, batch_idx):
        # Get patches from image
        x, _ = batch

        patches = self.to_patch(x)

        # Get batch size and number of patches
        batch, num_patches, *_ = patches.shape

        # Patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1 : (num_patches + 1)]

        # Calculate number of patches to mask
        num_masked = int(self.masking_ratio * num_patches)

        # Get random indices to choose random masked patches
        rand_indices = torch.rand(batch, num_patches).argsort(dim=-1).to(self.device)

        # Save masked and unmasked indices
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # Get the unmasked tokens to be encoded
        batch_range = torch.arange(batch)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # Get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # Attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # Project encoder to decoder dimensions, if they are not equal,
        # the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc2dec(encoded_tokens)

        # Reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # Repeat mask tokens for number of masked, and add the positions
        # using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # Concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # Splice out the mask tokens
        mask_tokens = decoded_tokens[:, :num_masked]

        # Project to pixel values
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        loss = F.mse_loss(pred_pixel_values, masked_patches)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    @property
    def backbone(self):
        return self.encoder

    def configure_optimizers(self):
        params = self.parameters()

        return _optimizer(params, self.config)
