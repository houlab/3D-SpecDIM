import torch.nn as nn
import torch
from utils.util import img_to_patch
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import math

from torch.autograd import Function
import numpy as np
from utils.util import plot_TSNE
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler

from typing import Callable, Optional

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MarginDisparityDiscrepancy(nn.Module):
    def __init__(self, source_disparity: Callable, target_disparity: Callable,
                 margin: Optional[float] = 4, reduction: Optional[str] = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.source_disparity = source_disparity
        self.target_disparity = target_disparity

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:

        source_loss = -self.margin * self.source_disparity(y_s, y_s_adv)
        target_loss = self.target_disparity(y_t, y_t_adv)
        if w_s is None:
            w_s = torch.ones_like(source_loss)
        source_loss = source_loss * w_s
        if w_t is None:
            w_t = torch.ones_like(target_loss)
        target_loss = target_loss * w_t

        loss = source_loss + target_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class RegressionMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
    def __init__(self, margin: Optional[float] = 1, loss_function=F.l1_loss, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            return loss_function(y_adv, y.detach(), reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            return loss_function(y_adv, y.detach(), reduction='none')

        super(RegressionMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin,
                                                                   **kwargs)


class InvLRScheduler(_LRScheduler):
    def __init__(self, optimizer, gamma, power, init_lr=0.1, weight_decay=0.0005, last_epoch=-1):
        self.gamma = gamma
        self.power = power
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * (1 + self.gamma * self.last_epoch) ** (-self.power) for base_lr in self.base_lrs]

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

        self.mlp_head_adv = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

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

    def forward(self, x, alpha):
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
        feature = x[0]
        
        class_output = self.mlp_head(feature)

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output_adv = self.mlp_head_adv(reverse_feature)

        return class_output, class_output_adv, feature
        
class MDD_ViT(L.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.loss_domain = torch.nn.MSELoss()
        self.tmp_epoch = 0
        self.mdd = RegressionMarginDisparityDiscrepancy(1.)
        # self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x, alpha):
        return self.model(x, alpha)

    def configure_optimizers(self):
        optimizer_dict = [{"params": filter(lambda p: p.requires_grad, self.model.input_layer.parameters()), "lr": 0.0001},
                  {"params": filter(lambda p: p.requires_grad, self.model.transformer.parameters()), "lr": 0.0001},
                  {"params": filter(lambda p: p.requires_grad, self.model.mlp_head.parameters()), "lr": 0.001}]
        optimizer = optim.SGD(optimizer_dict, lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
        
        param_lr = []
        for param_group in optimizer.param_groups:
            param_lr.append(param_group["lr"])
        scheduler = InvLRScheduler(optimizer, gamma=0.0001, power=0.75, init_lr=0.001, weight_decay=0.0005)
        

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'  # or 'epoch' for updating scheduler per epoch
            }
        }

    def _calculate_loss(self, batch, alpha, mode="train"):
        imgs, labels = batch

        data_source = imgs['source']
        preds_s, preds_s_adv, feature_s = self.model(data_source, alpha)
        # if torch.any(torch.isnan(feature_s)):
        #     for name, param in self.model.named_parameters():
        #         if param.grad is not None:
        #             print(name, param.grad.data.norm())
        preds_s = preds_s.view(-1)
        preds_s_adv = preds_s_adv.view(-1)
        gt_Centroid = labels['gt_centroids'].float()
        err_s_label = F.smooth_l1_loss(preds_s, gt_Centroid)
        
        data_target = imgs['target']
        preds_t, preds_t_adv, feature_t = self.model(data_target, alpha)
        preds_t = preds_t.view(-1)
        preds_t_adv = preds_t_adv.view(-1)

        mdd_loss = self.mdd(preds_s, preds_s_adv, preds_t, preds_t_adv)

        t_labels = np.array(labels['t_labels'])
        unique_filenames = np.unique(t_labels)
        spurious_label = torch.zeros_like(preds_t, dtype=torch.float32)
        for unique_filename in unique_filenames:
            mask = (t_labels == unique_filename)
            spurious_label[mask] = preds_t[mask].mean()

        err_t_label = F.smooth_l1_loss(preds_t, spurious_label)

        # preds_target_mean = torch.mean(preds_target)
        # val_loss = torch.mean((preds_target - preds_target_mean)**2)

        loss = err_s_label + err_t_label*0.5 - mdd_loss*0.01

        if self.tmp_epoch != self.current_epoch:
            tsne_s, tsne_t, similarity = plot_TSNE(feature_s, feature_t)

            logger = self.logger.experiment

            plt.figure()
            plt.scatter(tsne_s[:, 0], tsne_s[:, 1], label='source', color='red')
            plt.scatter(tsne_t[:, 0], tsne_t[:, 1], label='target', color='blue')
            plt.legend(loc='upper right')

            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()

            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image.convert('RGB'))
            image_array = image_array.astype(np.float32) / 255.0
            image_array = np.transpose(image_array, (2, 0, 1))
            image_tensor = torch.from_numpy(image_array)

            logger.add_image('TSNE_Fig', image_tensor, self.current_epoch)

            self.log("%s_similarity" % mode, similarity, sync_dist=True)

            self.tmp_epoch = self.current_epoch

        self.log("%s_loss_mdd" % mode, -mdd_loss*0.1, sync_dist=True)
        self.log("%s_loss" % mode, err_s_label, sync_dist=True)
        self.log("%s_loss_t" % mode, err_t_label, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        train_dataloader = self.trainer.train_dataloader
        len_dataloader = len(train_dataloader)
        n_epoch = self.trainer.max_epochs

        p = float(batch_idx + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        loss = self._calculate_loss(batch, alpha, mode="train")
        #TensorBoard
        # self.logger.experiment.add_scalar("train/loss", loss, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, 0, mode="val")

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        data_target = imgs['target'] 
        preds, _, _ = self.model(data_target, 0)
        preds = preds.view(-1) - 15

        for key, value in labels.items():
            if not isinstance(labels[key], list):
                labels[key] = value.cpu().float().numpy()
        labels['preds'] = preds.cpu().float().numpy()

        df = pd.DataFrame(labels)

        # logpath = self.trainer.default_root_dir
        filename = self.trainer.logger.log_dir.split('/')[-2]+'_outputs.csv'
        filepath = os.path.join(self.trainer.logger.log_dir, filename)

        if not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
        
        # self._calculate_loss(batch, mode="test")



