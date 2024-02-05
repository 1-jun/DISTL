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
# from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold


#%%
ROOT = '/home/wonjun/code/DISTL'
current_directory = Path(__file__).parent.absolute()
print(f"Current directory: {current_directory}")
seed = 2228

#%% MIMIC
mimic_path = '/home/wonjun/data/mimic-cxr-jpg-resized512'

metadata_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-metadata.csv')
mimic_split_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-split.csv')
negbio_labels_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-negbio.csv')
metadata = pd.read_csv(metadata_path)
mimic_split = pd.read_csv(mimic_split_path)
negbio_labels = pd.read_csv(negbio_labels_path)
negbio_labels = negbio_labels.fillna(0.0)
# negbio_labels = negbio_labels.replace(-1.0, 0.0) # -1.0 is for ambiguous menitoning of the lesion in the report


lesions = ['Atelectasis',
           'Cardiomegaly',
           'Consolidation',
           'Edema',
           'Enlarged Cardiomediastinum',
           'Fracture',
           'Lung Lesion',
           'Lung Opacity',
           'No Finding',
           'Pleural Effusion',
           'Pleural Other',
           'Pneumonia',
           'Pneumothorax',
           'Support Devices']


mimic_trainset_df = mimic_split[mimic_split['split']=='train']

# Use only frontal view CXRs (ie, ViewPosition is PA or AP)
df = pd.merge(mimic_trainset_df, metadata[['dicom_id', 'ViewPosition', 'StudyDate', 'StudyTime']],
              on='dicom_id', how='inner')
df = df[df['ViewPosition'].isin(['PA', 'AP'])]

# # Use only the first or last scans by StudyDate/StudyTime
# df['StudyDate'] = df['StudyDate'].astype(str)
# df['StudyTime'] = df['StudyTime'].astype(str)
# df['StudyDateTime'] = df['StudyDate'] + ' ' + df['StudyTime']


df = pd.merge(df, negbio_labels[['subject_id', 'study_id']+lesions],
              on=['subject_id', 'study_id'], how='left')
df = df.dropna()
# df = df[df[lesion] != -1] # -1 is for ambiguous mentions in the report; we'll just skip these for now

#%% CheXpert
chexpert_path = '/home/wonjun/data/CheXpert-v1.0'
df_train_path = os.path.join(chexpert_path, 'train.csv')
df = pd.read_csv(df_train_path)

lesions = ['Atelectasis',
           'Cardiomegaly',
           'Consolidation',
           'Edema',
           'Enlarged Cardiomediastinum',
           'Fracture',
           'Lung Lesion',
           'Lung Opacity',
           'No Finding',
           'Pleural Effusion',
           'Pleural Other',
           'Pneumonia',
           'Pneumothorax',
           'Support Devices']

df = df[df['AP/PA'].isin(['AP', 'PA'])]
df = df.fillna(0.0)
df
#%%
df['Cardiomegaly'].value_counts()

#%% all
# df.to_csv('mimic_multilabel_all.csv', index=False)



#%% Split into pretraining set and DISTL-training set
pretrain_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)

# Ideally, we would make the split so that all labels are evenly distributed
# but since that is very difficult, we just make the number of normals
# even between pretraining set and DISTL-training set
for fold0_indices, fold1_indices in pretrain_split.split(df, df['No Finding']):
    pretrain_set = df.iloc[fold1_indices]
    distl_set = df.iloc[fold0_indices]

# Split the DISTL set into three folds
folds = {}
kfold_split = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
for i, (_, fold_indices) in enumerate(kfold_split.split(distl_set, distl_set['No Finding'])):
    folds[f"fold_{i}"] = distl_set.iloc[fold_indices]

# save
pretrain_set.to_csv(os.path.join(current_directory, f"mimic_multilabel_labeled.csv"))
for fold_name, fold_df in folds.items():
    fold_df.to_csv(os.path.join(current_directory, f"mimic_multilabel_{fold_name}.csv"))
# %%
