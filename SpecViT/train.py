import os 
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from options import BaseOptions
from datasets.domain_spec_dataset import CustomDataModula
from models.create_models import create_model
# from models.ViT_skip import ViT

import time
from datetime import datetime

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

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print("Device:", device)

MyDataset = CustomDataModula(opt=opt)
MyDataset.setup()
train_loader = MyDataset.train_dataloader()
val_loader = MyDataset.val_dataloader()
test_loader = MyDataset.test_dataloader()

pos_manner = opt.pos_manner

Mymodel = create_model(opt)

def train_model(**kwargs):
    # create TensorBoardLogger，determine the savie_dir
    timestamp = time.time()
    formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
    log_name = f'{opt.model}-{pos_manner}'

    logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(CHECKPOINT_PATH, 'logs'), name=log_name, version=formatted_time)
    if opt.logger == 'False':
        logger = False

    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH),
        accelerator="auto",
        devices=opt.gpu_ids,
        max_epochs=120,
        logger=logger,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss",
            dirpath=CHECKPOINT_PATH, filename=f'{opt.model}-{pos_manner}'+'-{epoch:02d}-{val_loss:.2f}'),
            LearningRateMonitor("epoch"),
        ],
    )

    if opt.logger != 'False':
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, opt.pretrained_filename)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = Mymodel(**kwargs)
        model = load_pretrained_weights(model, pretrained_filename)

        trainer.fit(model, train_loader, val_loader)
        lastest_filename = os.path.join(CHECKPOINT_PATH, f'{opt.model}-{pos_manner}-latest.ckpt')
        trainer.save_checkpoint(lastest_filename)
        model = Mymodel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        # model = Mymodel.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = Mymodel(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = Mymodel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.validate(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["val_loss"], "val": val_result[0]["val_loss"]}
    return model, result


if __name__ == '__main__':
    model, results = train_model(
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
            "pos_manner": pos_manner,
        },
        lr=3e-4,
    )
    print("ViT results", results)



