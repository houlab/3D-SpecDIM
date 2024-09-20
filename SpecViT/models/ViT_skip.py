import torch.nn as nn
import torch
from utils.util import img_to_patch
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import math
from utils.irpe import get_rpe_config
from utils.irpe import build_rpe

class RPEAttention(nn.Module):
    '''
    Attention with image relative position encoding
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = \
            build_rpe(rpe_config,
                      head_dim=head_dim,
                      num_heads=num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)

        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0, pos_manner='learned'):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.pos_manner = pos_manner
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        if pos_manner == 'iRPE':
            rpe_config = get_rpe_config(
                ratio=1.9,
                method="product",
                mode='ctx',
                shared_head=True,
                skip=1,
                rpe_on='k',
            )
            self.attn = RPEAttention(embed_dim, num_heads=num_heads, qkv_bias=False, 
                                    qk_scale=None, attn_drop=0., proj_drop=0., 
                                    rpe_config=rpe_config)

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        if self.pos_manner == 'iRPE':
            inp_x = inp_x.transpose(0, 1)
            x = x.transpose(0,1)
            x = x + self.attn(inp_x)
            x = x.transpose(0, 1)
        else:
            x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout,
        pos_manner,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
            pos_manner - position embedding manner
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout, pos_manner=pos_manner) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = self.PosEmbedding(pos_manner, num_patches, embed_dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def PosEmbedding(self, pos_manner, num_patches=1, embed_dim=512):
        if pos_manner == 'sin': 
            pos_embedding = torch.zeros(1+num_patches, embed_dim)
            position = torch.arange(0, 1+num_patches).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
            pos_embedding[:, 0::2] = torch.sin(position * div_term)
            pos_embedding[:, 1::2] = torch.cos(position * div_term)
            pos_embedding = pos_embedding.unsqueeze(0)
        
        elif pos_manner in ['learned', 'iRPE']:
            pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

        else:  # default as learnable
            pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

        return pos_embedding
    
    def PixSpecMapping(self, pix):
        pix = pix - pix//2
        spec = 0.02215*pix**2+3.077*pix+591.6
        return spec

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1].to(x.device)

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

class ViT(L.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        # self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        milestones = [20, 40, 80, 100]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs).view(-1)
        preds = preds + labels['fitted_centroid'].float()

        gt_Centroid = labels['gt_centroids'].float()
        loss = F.smooth_l1_loss(preds, gt_Centroid)
        self.log("%s_loss" % mode, loss, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        #TensorBoard
        # self.logger.experiment.add_scalar("train/loss", loss, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).view(-1)
        preds = preds + labels['fitted_centroid'].float()
        for key, value in labels.items():
            labels[key] = value.cpu().float().numpy()
        labels['preds'] = preds.cpu().float().numpy()

        df = pd.DataFrame(labels)

        filename = self.trainer.logger.log_dir.split('/')[-2]+'_outputs.csv'
        filepath = os.path.join(self.trainer.logger.log_dir, filename)
        
        if not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
        
        # self._calculate_loss(batch, mode="test")



