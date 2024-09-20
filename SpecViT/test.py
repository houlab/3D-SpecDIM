import os 
import lightning as L
import torch

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl

from options import BaseOptions
from datasets.test_dataset import CustomDataModula
from models.create_models import create_model

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


plt.set_cmap("cividis")
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

torch.set_float32_matmul_precision('medium')

opt = BaseOptions().parse()
opt.dataroot = opt.dataroot 

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = opt.dataroot 
# Path to the folder where the pretrained models are saved
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

pretrained_filename = opt.pretrained_filename

version = pretrained_filename
epoch_index = version.find("epoch")
# 提取 "epoch" 之后的部分，并进行替换
version = version[epoch_index:-len(".ckpt")].replace("-", "_").replace("=", "_").replace(".", "_")


log_name = f'{opt.model}-{opt.pos_manner}'
logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(DATASET_PATH, 'outputs'), 
                                      name=log_name, version=version)

Mymodel = create_model(opt)

def test_model(**kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(DATASET_PATH),
        accelerator="auto",
        devices=opt.gpu_ids,
        logger=logger,
    )

    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_file = os.path.join(CHECKPOINT_PATH, pretrained_filename)
    if os.path.isfile(pretrained_file):
        print("Found pretrained model at %s, loading..." % pretrained_file)
        model = Mymodel(**kwargs)
        model = load_pretrained_weights(model, pretrained_file)
    else:
        print("There is no pretrained model at %s, please check it!"% pretrained_file)


    trainer.test(model, dataloaders=test_loader, verbose=False)

    return model

if __name__ == '__main__':
    model = test_model(
        model_kwargs={
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
        lr=3e-4,
    )




