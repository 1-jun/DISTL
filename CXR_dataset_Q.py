#%%
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms as pth_transforms

import utils

seed = 2228

#%%
class MIMIC_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 mimic_path,
                 lesion,
                 transforms,
                 total_folds,
                 mode,
                 labeled):
        self.mimic_path = mimic_path
        self.lesion = lesion
        self.transforms = transforms
        
        self.mode = mode
        self.labeled = labeled
        
        if total_folds == 0:
            self.total_folds = ['labeled']
            self.pseudo_folds = None
        elif total_folds == 1:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0']
        elif total_folds == 2:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0', 'fold_1']
        elif total_folds == 3:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0', 'fold_1', 'fold_2']
        self.test_fold = ['test']
        
        dfs = []
        if self.mode == 'train':
            if self.labeled:
                for fold in self.total_folds:
                    df_path = os.path.join(data_path, f"mimic_{lesion.replace(' ', '')}_{fold}.csv")
                    dfs.append(pd.read_csv(df_path))
            elif not self.labeled:
                for fold in self.pseudo_folds:
                    df_path = os.path.join(data_path, f"mimic_{lesion.replace(' ', '')}_{fold}.csv")
                    dfs.append(pd.read_csv(df_path))
        self.df = pd.concat(dfs, axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.mimic_path, 'files',
                                f"p{str(row.subject_id)[:2]}",
                                f"p{str(row.subject_id)}",
                                f"s{str(row.study_id)}",
                                f"{row.dicom_id}.jpg")
        image = cv2.imread(img_path, 1)
        image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        
        # Apply DINO Augmentation
        images = self.transforms(image) 
        
        if self.labeled:
            label = int(row[self.lesion])
        else:
            label = []
        
        return images, label
    
#%%
class MIMIC_Multilabel_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 mimic_path,
                 lesions,
                 transforms,
                 total_folds,
                 mode,
                 labeled):
        self.mimic_path = mimic_path
        self.lesions = lesions
        self.transforms = transforms
        
        self.mode = mode
        self.labeled = labeled
        
        if total_folds == 0:
            self.folds = None
        elif total_folds == 1:
            self.folds = ['fold_0']
        elif total_folds == 2:
            self.folds = ['fold_0', 'fold_1']
        elif total_folds == 3:
            self.folds = ['fold_0', 'fold_1', 'fold_2']
        self.test_fold = ['test']
        
        dfs = []
        if self.mode == 'train':
            if self.labeled:
                df_path = os.path.join(data_path, f"labeled.csv")
                dfs.append(pd.read_csv(df_path))
            elif not self.labeled:
                for fold in self.folds:
                    df_path = os.path.join(data_path, f"{fold}.csv")
                    dfs.append(pd.read_csv(df_path))
        self.df = pd.concat(dfs, axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.mimic_path, 'files',
                                f"p{str(row.subject_id)[:2]}",
                                f"p{str(row.subject_id)}",
                                f"s{str(row.study_id)}",
                                f"{row.dicom_id}.jpg")
        image = cv2.imread(img_path, 1)
        image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        
        # Apply DINO Augmentation
        images = self.transforms(image) 
        
        if self.labeled:
            label = torch.Tensor(row[self.lesions].values.astype(int))
        else:
            label = []
        
        return images, label
    
#%%
class MIMIC_Multilabel_All_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 mimic_path,
                 lesions,
                 transforms,
                 mode,
                 labeled,
                 subset_proportion):
        self.mimic_path = mimic_path
        self.lesions = lesions
        self.transforms = transforms
        
        self.mode = mode
        self.labeled = labeled
        
        df = pd.read_csv(data_path)
        n_samples = int(len(df) * subset_proportion)
        self.df = df.sample(n_samples, random_state=seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.mimic_path, 'files',
                                f"p{str(row.subject_id)[:2]}",
                                f"p{str(row.subject_id)}",
                                f"s{str(row.study_id)}",
                                f"{row.dicom_id}.jpg")
        image = cv2.imread(img_path, 1)
        image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        
        # Apply DINO Augmentation
        images = self.transforms(image) 
        
        if self.labeled:
            label = torch.Tensor(row[self.lesions].values.astype(int))
        else:
            label = []
        
        return images, label
            
#%%
class MIMIC_Staggered(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 mimic_path,
                 lesions,
                 transforms,
                 fold):
        self.mimic_path = mimic_path
        self.lesions = lesions
        self.transforms = transforms
        
        
        dfs = []
        df_path = os.path.join(data_path, f"mimic_multilabel_labeled.csv")
        dfs.append(pd.read_csv(df_path))

        for i in range(fold+1):
            df_path = os.path.join(data_path, f"mimic_multilabel_fold_{i}.csv")
            dfs.append(pd.read_csv(df_path))

        self.df = pd.concat(dfs, axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.mimic_path, 'files',
                                f"p{str(row.subject_id)[:2]}",
                                f"p{str(row.subject_id)}",
                                f"s{str(row.study_id)}",
                                f"{row.dicom_id}.jpg")
        image = cv2.imread(img_path, 1)
        image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        
        # Apply DINO Augmentation
        images = self.transforms(image) 
        
        label = torch.Tensor(row[self.lesions].values.astype(int))
        
        return images, label
