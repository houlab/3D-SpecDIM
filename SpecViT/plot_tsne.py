import os 
import lightning as L
import torch

from options import BaseOptions
from domain_spec_dataset import CustomDataModula
from models.create_models import create_model
from utils.util import plot_TSNE
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

def load_pretrained_weights(new_model, pretrained_model):
    # 获取预训练模型的状态字典
    checkpoint = torch.load(pretrained_model)

    # 从 checkpoint 字典中获取 state_dict
    pretrained_state_dict = checkpoint['state_dict']
    new_state_dict = new_model.state_dict()

    # 从预训练模型的状态字典中提取与新模型匹配的部分
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in new_state_dict}

    # 更新新模型的状态字典
    new_state_dict.update(pretrained_state_dict)
    new_model.load_state_dict(new_state_dict)
    return new_model

torch.set_float32_matmul_precision('medium')

opt = BaseOptions().parse()
opt.dataroot = opt.dataroot 

DATASET_PATH = opt.dataroot 
CHECKPOINT_PATH = opt.checkpoints_dir

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
gpu_ids = opt.gpu_ids

MyDataset = CustomDataModula(opt=opt)
MyDataset.setup(stage='Test')
test_loader = MyDataset.test_dataloader()

pretrained_file = os.path.join(CHECKPOINT_PATH, opt.pretrained_filename)

Mymodel = create_model(opt)

if os.path.isfile(pretrained_file):
    print("Found pretrained model at %s, loading..." % pretrained_file)
    # Automatically loads the model with the saved hyperparameters
    # model = Mymodel.load_from_checkpoint(pretrained_file)
    model = Mymodel(model_kwargs={
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "patch_size": 4,
        "num_channels": 2,
        "num_patches": 256,
        "num_classes": 1,
        "dropout": 0.2,
        "pos_manner": 'learned',
    },
    lr=3e-4,)
    model = load_pretrained_weights(model, pretrained_file)
else:
    print("There is no pretrained model at %s, please check it!"% pretrained_file)
model.eval()

log_name = f'{opt.model}-{opt.pos_manner}'
save_dir=os.path.join(CHECKPOINT_PATH, 'logs', log_name)

i = 0
features_s = None
with torch.no_grad():
    for batch in test_loader:
        i = i+1
        if i >50:
            continue
        imgs, labels = batch
        alpha = 0
        data_source = imgs['source']
        preds, domain_output, feature_s = model(data_source, alpha)
        preds = preds.view(-1)
        gt_Centroid = labels['gt_centroids'].float()

        data_target = imgs['target']
        preds_t, domain_output, feature_t = model(data_target, alpha)
        preds_t = preds_t.view(-1)

        if features_s is None:
            features_s = feature_s
            features_t = feature_t
        else:
            features_s = torch.concatenate((features_s, feature_s), axis=0)
            features_t = torch.concatenate((features_t, feature_t), axis=0)

    tsne_s, tsne_t, similarity = plot_TSNE(features_s, features_t)

    tsne_s = tsne_s[::10]
    tsne_t = tsne_t[::10]


    plt.figure()
    plt.scatter(tsne_s[:, 0], tsne_s[:, 1], label='source', color='red')
    plt.scatter(tsne_t[:, 0], tsne_t[:, 1], label='target', color='blue')
    plt.xlim([-25, 25])  # x轴范围从0到6
    plt.ylim([-25, 25])  # y轴范围从0到12
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(save_dir, f'tsne_plot_{np.array(similarity).astype(int)}.png'))
    plt.close()




