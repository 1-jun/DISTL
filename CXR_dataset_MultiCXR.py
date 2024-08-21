import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms as pth_transforms

# import utils

seed = 2228

class MultiCXR_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 categories,
                 transforms,
                #  total_folds,
                #  mode,
                 labeled):
        self.categories = categories
        self.transforms = transforms
        
        # self.mode = mode
        self.labeled = labeled
        
        # if total_folds == 0:
        #     self.folds = None
        # elif total_folds == 1:
        #     self.folds = ['fold_0']
        # elif total_folds == 2:
        #     self.folds = ['fold_0', 'fold_1']
        # elif total_folds == 3:
        #     self.folds = ['fold_0', 'fold_1', 'fold_2']
        # self.test_fold = ['test']
        # dfs = []
        # if self.mode == 'train':
        #     if self.labeled:
        #         df_path = os.path.join(data_path, f"labeled.csv")
        #         dfs.append(pd.read_csv(df_path))
        #     elif not self.labeled:
        #         for fold in self.folds:
        #             df_path = os.path.join(data_path, f"{fold}.csv")
        #             dfs.append(pd.read_csv(df_path))
        # self.df = pd.concat(dfs, axis=0)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # img_path = os.path.join(self.mimic_path, 'files',
        #                         f"p{str(row.subject_id)[:2]}",
        #                         f"p{str(row.subject_id)}",
        #                         f"s{str(row.study_id)}",
        #                         f"{row.dicom_id}.jpg")
        img_path = row['img_path']
        image = cv2.imread(img_path, 1) # 1 flag is for cv2.IMREAD_COLOR
        image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)
        
        # Apply DINO Augmentation
        images = self.transforms(image) 
        
        if self.labeled:
            label = torch.Tensor(row[self.categories].values.astype(int))
        else:
            label = []
        
        return images, label