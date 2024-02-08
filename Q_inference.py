#%%
import os#; os.environ['CUDA_VISIBLE_DEVICES']='3'
import argparse
import cv2
import random
import colorsys
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

# import skimage.io
# from skimage.measure import find_contours
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vit_o
import glob

import sklearn.metrics

# # combine consolidation and pnumonia to create a new column
# def combine_cnsl_pna(row):
#     if row['Consolidation']==1 or row['Pneumonia']==1:
#         return 1.0
#     elif row['Consolidation']==-1 or row['Pneumonia']==-1:
#         return -1.0
#     else:
#         return 0.0

#%% Load CheXpert
# root = Path('/media/wonjun/HDD2TB/CheXpert-v1.0')
# df_train = pd.read_csv(os.path.join(root, "train.csv"))
# df_train = df_train.fillna(0.0)
# df_valid = pd.read_csv(os.path.join(root, "valid.csv"))
# df_train = df_train[df_train['Frontal/Lateral']=='Frontal']
# df_valid = df_valid[df_valid['Frontal/Lateral']=='Frontal']
# df_train['Cnsl_Pna'] = df_train.apply(combine_cnsl_pna, axis=1)
# df_valid['Cnsl_Pna'] = df_valid.apply(combine_cnsl_pna, axis=1)
# df = df_train[df_train['Cnsl_Pna']==1]
# # df = df_train[df_train['Pneumothorax']==1]
# display(df)

#%% Load MIMIC val set
mimic_path = Path('/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512')

metadata_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-metadata.csv')
mimic_split_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-split.csv')
negbio_labels_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-negbio.csv')
metadata = pd.read_csv(metadata_path)
mimic_split = pd.read_csv(mimic_split_path)
negbio_labels = pd.read_csv(negbio_labels_path)
negbio_labels = negbio_labels.fillna(0.0)

labels = ['Atelectasis',
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
label_to_ind = {l:i for i,l in enumerate(labels)}
ind_to_label = {i:l for i,l in enumerate(labels)}

test_df = mimic_split[mimic_split['split']=='train']

# Use only frontal view CXRs (ie, ViewPosition is PA or AP)
df = pd.merge(test_df,
              metadata[['dicom_id', 'ViewPosition', 'StudyDate', 'StudyTime']],
              on='dicom_id', how='inner')
df = df[df['ViewPosition'].isin(['PA', 'AP'])]

# # Use only the first or last scans by StudyDate/StudyTime
# df['StudyDate'] = df['StudyDate'].astype(str)
# df['StudyTime'] = df['StudyTime'].astype(str)
# df['StudyDateTime'] = df['StudyDate'] + ' ' + df['StudyTime']


df = pd.merge(df, negbio_labels[['subject_id', 'study_id']+labels],
              on=['subject_id', 'study_id'], how='left')
df = df.dropna()
display(df)

#%%
def load_img(img_path, img_size=(224,224), patch_size=8):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(img)
    img = pth_transforms.Compose(
        [
            utils.GaussianBlurInference(),
            pth_transforms.ToTensor()
        ]
    )(img) # [3, 224, 224]
    
    # make the image divisible by patch size
    w, h = img.shape[1]-img.shape[1]%patch_size, img.shape[2]-img.shape[2]%patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    return img, w_featmap, h_featmap


#%%
# PRETRAINED_WEIGHTS = 'outputs/mimic-split-10-30-30-30/mimic_multilabel_fold2/checkpoint.pth'
PRETRAINED_WEIGHTS = 'outputs/mimic_multilabel_all_mainrun/checkpoint.pth'
CHECKPOINT_KEY = 'student'
PATCH_SIZE, OUT_DIM, NUM_CLASSES = 8, 65536, len(labels)

model = vit_o.__dict__['vit_small'](patch_size=PATCH_SIZE, num_classes=0)
embed_dim = model.embed_dim
model = utils.MultiCropWrapper(
    model,
    vit_o.DINOHead(in_dim=embed_dim, out_dim=OUT_DIM),
    vit_o.CLSHead(in_dim=384, hidden_dim=256, num_classes=NUM_CLASSES)
)
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.to('cuda')

state_dict = torch.load(PRETRAINED_WEIGHTS, map_location='cpu')
state_dict = state_dict[CHECKPOINT_KEY]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
msg = model.load_state_dict(state_dict, strict=True)
pprint('Pretrained weights found at {} and loaded with msg: {}'.format(PRETRAINED_WEIGHTS, msg))


#%%
print(f"Results for {PRETRAINED_WEIGHTS}")
preds = []
for i, row in tqdm(df.iterrows(), total=len(df), desc='Predicting...', colour='green'):
    img_path = os.path.join(mimic_path, 'files',
                            f"p{str(row['subject_id'])[:2]}",
                            f"p{str(row['subject_id'])}",
                            f"s{str(row['study_id'])}",
                            row['dicom_id']+'.jpg')

    img, w_featmap, h_featmap = load_img(
        img_path=img_path, img_size=(224,224)
    )
    img = img.to('cuda')

    pred = model(img)
    pred = [p.detach().cpu().numpy()[0][0] for p in pred]
    pred = [1 if p>=0.5 else 0 for p in pred]
    preds.append(pred)

results = []
for label in labels:
    ind = label_to_ind[label]
    preds_for_label = np.array(preds)[:, ind]
    trues_for_label = df[labels].values[:, ind]

    mask = trues_for_label != -1
    _tn, _fp, _fn, _tp = sklearn.metrics.confusion_matrix(preds_for_label[mask], trues_for_label[mask]).ravel()
    
    eps = 3e-9 # because python complains when one of tn/fp/fn/tp is 0
    tn, fp, fn, tp = _tn+eps, _fp+eps, _fn+eps, _tp+eps
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision*recall) / (precision+recall)
    
    # print(f"{label.upper()}")
    # print(f"Precision: {precision*100:.4f}%")
    # print(f"Recall: {recall*100:.4f}%")
    # print(f"F1: {f1:.4f}")
    # print()
    
    result = {"tn":_tn,
              "fp":_fp,
              "fn":_fn,
              "tp":_tp,
              "Precision":precision,
              "Recall":recall,
              "F1":f1}
    results.append(result)

all_tn = sum([d["tn"] for d in results])
all_fp = sum([d["fp"] for d in results])
all_fn = sum([d["fn"] for d in results])
all_tp = sum([d["tp"] for d in results])
all_precision = all_tp / (all_tp + all_fp)
all_recall = all_tp / (all_tp + all_fn)
all_f1 = 2 * (all_precision*all_recall)/(all_precision + all_recall)
all_results = {"tn": all_tn,
               "fp": all_fp,
               "fn": all_fn,
               "tp": all_tp,
               "Precision": all_precision,
               "Recall": all_recall,
               "F1": all_f1}
results.append(all_results)

results_df = pd.DataFrame(results).round(4)
results_df.index = labels + [""]
display(results_df)

# %%
y_true = np.array([[0,1,1],
                    [1,0,1]])
y_pred = np.array([[0,1,1],
                    [1,1,1]])

indices_to_skip = [(1,1)]
mask = np.ones_like(y_true, dtype=bool)
for i,j in indices_to_skip:
    mask[i,j] = False

y_true = y_true[mask]
y_pred = y_pred[mask]

tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
print(tn, fp, fn, tp)

#%%
