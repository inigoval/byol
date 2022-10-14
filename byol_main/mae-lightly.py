import torch
import torch.nn as nn
from lightly.models.modules import masked_autoencoder
import lightly.models.utils as utils

from byol_main.utilities import _optimizer
from byol_main.evaluation import Lightning_Eval
from networks.models import _get_transformer


# Yoinked from lightly
# https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/masked_autoencoder.py
class MAE(Lightning_Eval):
    def __init__(self, config):
        super().__init__(config)

        # Get number of channels
        vit = _get_transformer(config)

        # Adjust patching layer based on number of in_channels
        # vit.conv_proj = nn.Conv2d(
        #     in_channels=config["data"]["color_channels"],
        #     out_channels=config["model"]["features"],
        #     kernel_size=config["model"]["patch_size"],
        #     stride=config["model"]["patch_size"],
        # )
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config["model"]["decoder_dim"]))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=config["model"]["decoder_dim"],
            mlp_dim=config["model"]["decoder_dim"] * 4,
            out_dim=vit.patch_size**2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()

    # TODO check what idx_keep does
    def forward(self, x, idx_keep=None):
        return self.backbone(x, idx_keep)

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(self.mask_token, (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode)

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        images, _ = batch

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        return loss

    def configure_optimizers(self):
        params = self.parameters()
        return _optimizer(params, self.config)
