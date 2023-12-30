#%%
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms as pth_transforms

import utils

#%%
class MIMIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, df_path, mimic_path, lesion):
        self.df = pd.read_csv(df_path)
        self.mimic_path = mimic_path
        self.lesion = lesion

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
        if not self.transforms == None:
            images = self.transforms(images)
        
        # Apply NO augmentation
        elif self.transforms == None:
            image = pth_transforms.Compose(
                [utils.GaussianBlurInference(),
                 pth_transforms.ToTensor()]
            )(image)
            
        label = int(row[self.lesion])
        
        return image, label
            
        