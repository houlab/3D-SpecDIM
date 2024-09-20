from torchvision import transforms

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import tifffile
from pytorch_lightning import LightningDataModule
import torch 
from PIL import Image
import glob

class CustomDataset(Dataset):
    def __init__(self, opt, transforms=None):
        self.opt = opt
        # load raw image data 
        dataroot = opt.dataroot
        self.transforms = transforms

        source_file_folder = os.path.join(dataroot,'source')
        source_file = glob.glob(os.path.join(source_file_folder, '*.tiff'))
        self.s_num = len(source_file) * opt.N_per_C

        self.s_imgs = np.zeros((self.s_num, 16, 80))
        self.s_labels = np.zeros(self.s_num)
        
        for i, file_name in enumerate(source_file):
            self.s_imgs[i*opt.N_per_C:(i+1)*opt.N_per_C,:,:] = tifffile.imread(file_name)
            start = file_name.find("centroid") + len("centroid")
            end = file_name.find(".tiff")
            label_centroid = float(file_name[start:end].replace('_','.'))
            self.s_labels[i*opt.N_per_C:(i+1)*opt.N_per_C] = label_centroid

        random_indices = np.random.permutation(len(self.s_imgs))
        self.s_imgs = self.s_imgs[random_indices,:,:]
        self.s_labels = self.s_labels[random_indices]

        targe_file_folder = os.path.join(dataroot,'target')
        target_file = glob.glob(os.path.join(targe_file_folder, '*.tiff'))
        
        self.t_imgs = None
        self.t_labels = np.array([],dtype=str)
        for file_name in target_file:
            t_img = tifffile.imread(file_name)
            if self.t_imgs is None:
                self.t_imgs = t_img
            else:
                self.t_imgs = np.concatenate((self.t_imgs, t_img), axis=0)
            
            t_labels = np.array([file_name.split('/')[-1]]*len(t_img),dtype=str)
            self.t_labels = np.concatenate((self.t_labels, t_labels))

        self.t_num = self.t_imgs.shape[0]

        df = pd.read_csv(os.path.join(source_file_folder,'labels.csv'))
        self.data_dict = df.to_dict('list')

    def __len__(self):
        return min(self.s_num, self.t_num)
    
    def __getitem__(self,idx):
        s_img = self.s_imgs[idx,:,:]
        t_img = self.t_imgs[idx,:,:]

        s_img = s_img[:,:,np.newaxis] # [16x80x1]
        t_img = t_img[:,:,np.newaxis]

        psf_curve = np.mean(s_img[:,0:16],axis=0,keepdims=True).astype('float32')
        spec_curve = np.mean(s_img[:,16:],axis=0,keepdims=True).astype('float32')
        psf_curve = (psf_curve - np.min(psf_curve)) / (np.max(psf_curve) - np.min(psf_curve))
        spec_curve = (spec_curve - np.min(spec_curve)) / (np.max(spec_curve) - np.min(spec_curve))

        s_spec = np.concatenate((psf_curve, spec_curve), 1)
        # spec = ((spec - np.min(spec)) / (np.max(spec) - np.min(spec)))

        psf_curve = np.mean(t_img[:,0:16],axis=0,keepdims=True).astype('float32')
        spec_curve = np.mean(t_img[:,16:],axis=0,keepdims=True).astype('float32')
        psf_curve = (psf_curve - np.min(psf_curve)) / (np.max(psf_curve) - np.min(psf_curve))
        spec_curve = (spec_curve - np.min(spec_curve)) / (np.max(spec_curve) - np.min(spec_curve))

        t_spec = np.concatenate((psf_curve, spec_curve), 1)

        if self.transforms:
            # img = torch.from_numpy(img)
            s_img = Image.fromarray((s_img * 255).astype(np.uint8).squeeze())
            t_img = Image.fromarray((t_img * 255).astype(np.uint8).squeeze())
            s_img = self.transforms(s_img)
            t_img = self.transforms(t_img)

        s_spec = torch.tensor(np.repeat(s_spec,s_img.shape[1],axis=0))
        s_img = torch.cat((s_img, s_spec.permute(2,0,1)),dim=0)

        t_spec = torch.tensor(np.repeat(t_spec,t_img.shape[1],axis=0))
        t_img = torch.cat((t_img, t_spec.permute(2,0,1)),dim=0)
     
        labels = {}
        labels['gt_centroids'] = self.s_labels[idx]
        labels['t_labels'] = self.t_labels[idx]

        img = {}
        img['source'] = s_img
        img['target'] = t_img

        return img, labels


class CustomDataModula(LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.batch_size = opt.batchSize 

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # For training, we add some augmentation. Networks are too powerful and would overfit.
        self.train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def setup(self, stage='Train'):
        if stage == 'Test':
            self.dataset = CustomDataset(self.opt, transforms = self.test_transform)
            self.test_dataset = self.dataset

        elif stage == 'Train':
            self.dataset = CustomDataset(self.opt, transforms = self.train_transform)
            train_size = int(0.8 * len(self.dataset))
            val_size = int(0.1 * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4)

