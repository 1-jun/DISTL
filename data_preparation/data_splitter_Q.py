#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob
import cv2
from tqdm import tqdm
import random
import argparse
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
#%%
ROOT = '/root/repos/DISTL'
current_directory = Path(__file__).parent.absolute()
seed = 2228
#%%
# mimic_path = '/media/wonjun/HDD8TB/MIMIC-CXR-JPG-Resized512'
mimic_path = '/mnt/e/data/MIMIC-CXR-JPG-mini'
#%%'
metadata_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-metadata.csv')
mimic_split_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-split.csv')
negbio_labels_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-negbio.csv')
metadata = pd.read_csv(metadata_path)
mimic_split = pd.read_csv(mimic_split_path)
negbio_labels = pd.read_csv(negbio_labels_path)
negbio_labels = negbio_labels.fillna(0.0)
# negbio_labels = negbio_labels.replace(-1.0, 0.0) # -1.0 is for ambiguous menitoning of the lesion in the report

lesion = 'Pleural Effusion'

#%%
mimic_trainset_df = mimic_split[mimic_split['split']=='train']

# Use only frontal view CXRs (ie, ViewPosition is PA or AP)
df = pd.merge(mimic_trainset_df, metadata[['dicom_id', 'ViewPosition', 'StudyDate', 'StudyTime']],
              on='dicom_id', how='inner')
df = df[df['ViewPosition'].isin(['PA', 'AP'])]

# # Use only the first or last scans by StudyDate/StudyTime
# df['StudyDate'] = df['StudyDate'].astype(str)
# df['StudyTime'] = df['StudyTime'].astype(str)
# df['StudyDateTime'] = df['StudyDate'] + ' ' + df['StudyTime']


df = pd.merge(df, negbio_labels[['subject_id', 'study_id', lesion]],
              on=['subject_id', 'study_id'], how='left')
df = df.dropna()

#%% Split into pretraining set and DISTL-training set
pretrain_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
for fold0_indices, fold1_indices in pretrain_split.split(df, df[lesion]):
    pretrain_set = df.iloc[fold1_indices]
    distl_set = df.iloc[fold0_indices]

# %% Split the DISTL set into three folds
folds = {}
kfold_split = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
for i, (_, fold_indices) in enumerate(kfold_split.split(distl_set, distl_set[lesion])):
    folds[f"fold{i}"] = distl_set.iloc[fold_indices]

#%%
pretrain_set.to_csv(os.path.join(current_directory, f"mimic_{lesion.replace(' ','')}_pretrain.csv"))
for fold_name, fold_df in folds.items():
    fold_df.to_csv(os.path.join(current_directory, f"mimic_{lesion.replace(' ','')}_{fold_name}.csv"))