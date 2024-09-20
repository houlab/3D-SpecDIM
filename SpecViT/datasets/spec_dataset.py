from torchvision import transforms

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import tifffile
from pytorch_lightning import LightningDataModule
import torch 
from PIL import Image

from scipy.optimize import curve_fit

class CustomDataset(Dataset):
    def __init__(self, opt, transforms=None):
        self.opt = opt
        # load raw image data 
        dataroot = opt.dataroot

        if 'experiments' in dataroot:
            self.dataMode = 'real'
        else:
            self.dataMode = 'simu'

        file_extension = '.tiff'
        self.file_names = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.endswith(file_extension)]
        
        if self.dataMode == 'simu':
            # load labels
            self.N_per_C = opt.N_per_C
            df = pd.read_csv(dataroot+'/labels.csv')
            # Convert DataFrame to a dictionary
            self.data_dict = df.to_dict('list')
            self.transforms = transforms

        elif self.dataMode == 'real':
            imgs = tifffile.imread(self.file_names[0])
            self.N_per_C = imgs.shape[0]
            self.transforms = transforms

    def __len__(self):
        if self.dataMode == 'simu':
            return len(self.file_names * self.N_per_C)
        elif self.dataMode == 'real':
            return len(self.file_names * self.N_per_C)
        
    def PixSpecMapping(self, pix):
        spec = 0.02215*pix**2+3.077*pix+591.6
        return spec
    
    # 定义正态分布函数
    def modelFun(self, x, p1, p2, p3):
        return p1*np.exp(-((x-p2)/p3)**2)
        # return p(1) * exp(-((x - p(2)) / p(3)).^2);
    
    def __getitem__(self,idx):
        fileid = idx // self.N_per_C
        imgs = tifffile.imread(self.file_names[fileid])

        N_in_C = idx - fileid * self.N_per_C
        img = imgs[N_in_C,:,:] # random crop
        img = img[:,:,np.newaxis] # [16x80x1]
        
        psf_curve = np.mean(img[:,0:16],axis=0,keepdims=True).astype('float32')
        spec_curve = np.mean(img[:,16:],axis=0,keepdims=True).astype('float32')
        psf_curve = (psf_curve - np.min(psf_curve)) / (np.max(psf_curve) - np.min(psf_curve))
        spec_curve = (spec_curve - np.min(spec_curve)) / (np.max(spec_curve) - np.min(spec_curve))

        # img = ((img - np.min(img)) / (np.max(img) - np.min(img))).astype('float32')

        spec = np.concatenate((psf_curve, spec_curve),1)
        # spec = ((spec - np.min(spec)) / (np.max(spec) - np.min(spec)))

        fitted_centroid = 0

        # try:
        #     x = np.arange(0,16,1)
        #     startingVals = [1, 8, 0.5]
        #     popt, _ = curve_fit(self.modelFun, x, psf_curve.squeeze(), p0=startingVals, maxfev=10000)
        #     Pixpos = round(popt[1], 4)

        #     x = np.arange(0,64,1)
        #     startingVals = [1, 32, 10]            
        #     popt, _ = curve_fit(self.modelFun, x, spec_curve.squeeze(), p0=startingVals, maxfev=10000)
        #     Pixspec = round(popt[1], 4) + 16

        #     if Pixpos > 11 or Pixpos < 5 or Pixspec > 70 or Pixspec < 20 \
        #         or Pixspec - Pixpos > 70 or Pixspec - Pixpos < 0:
        #         Pixspec = np.argmax(spec_curve, axis=1)+16
        #         Pixpos  = np.argmax(psf_curve, axis=1)

        # except:
        #     # fitted_centroid = np.argmax(spec, axis=1)
        #     Pixspec = np.argmax(spec_curve, axis=1)[0,0]+16
        #     Pixpos  = np.argmax(psf_curve, axis=1)[0,0]

        # pix_dis = Pixspec - Pixpos

        # if Pixspec >=16 and Pixspec < 40:
        #     fitted_centroid = self.PixSpecMapping(pix_dis-42)
        # elif Pixspec >=40 and Pixspec < 60:
        #     fitted_centroid = self.PixSpecMapping(pix_dis-35)
        # elif Pixspec >=60 and Pixspec < 80:
        #     fitted_centroid = self.PixSpecMapping(pix_dis-33)
        # else:
        #     # fitted_centroid = np.argmax(spec, axis=1)
        #     fitted_centroid = 0


        if self.transforms:
            # img = torch.from_numpy(img)
            img = Image.fromarray((img * 255).astype(np.uint8).squeeze())
            img = self.transforms(img)

        spec = torch.tensor(np.repeat(spec,img.shape[1],axis=0))
        img = torch.cat((img, spec.permute(2,0,1)),dim=0)

        if self.dataMode == 'simu':
            start = self.file_names[fileid].find("centroid") + len("centroid")
            end = self.file_names[fileid].find(".tiff")
            label_centroid = float(self.file_names[fileid][start:end].replace('_','.'))
            
            ind = np.argwhere(np.array(self.data_dict['centroid']) == label_centroid)
            ind = ind.squeeze()[0] + N_in_C
            labels = {}
            labels['gt_centroids'] = label_centroid
            labels['photons'] = self.data_dict['photons'][ind]
            labels['background'] = self.data_dict['background'][ind]
            labels['diffusivity'] = self.data_dict['diffusivity'][ind]
            labels['fitted_centroid'] = fitted_centroid
        
        elif self.dataMode == 'real':
            labels = {}
            labels['fitted_centroid'] = fitted_centroid

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

