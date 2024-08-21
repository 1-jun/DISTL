#%%
import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn.preprocessing import MultiLabelBinarizer

categories = ['atelectasis', 'cardiomegaly', 'consolidation',
       'edema', 'effusion', 'enlarged_cardiomediastinum',
       'nodule', 'opacity',
       'pleural_other', 'pneumothorax']

#%% SNU-CXR to Multi-CXR
snu_path = '/home/wonjun/data/SNU-CXR/train'
files = glob.glob(f"{snu_path}/**/*", recursive=True)
files = [s for s in files if s.endswith('.png')]

new_entries = []
for file in files:
    new_entry = {}
    new_entry['dataset'] = 'snu'
    new_entry['img_path'] = file
    new_entry.update({k:0 for k in categories})
    if 'nodule' in file:
        new_entry['nodule'] = 1
    if 'pneumonia' in file:
        new_entry['consolidation'] = 1
    if 'pneumothorax' in file:
        new_entry['pneumothorax'] = 1
    new_entries.append(new_entry)
new_snu_df = pd.DataFrame(new_entries)
# new_snu_df[categories].sum()

# %% MIMIC to Multi-CXR
mimic_path = '/home/wonjun/data/mimic-cxr-jpg-resized512'

mimic_metadata = pd.read_csv(os.path.join(mimic_path, 'mimic-cxr-2.0.0-metadata.csv'))
mimic_metadata = mimic_metadata[mimic_metadata['ViewPosition'].isin(['PA', 'AP'])]

mimic_split = pd.read_csv(os.path.join(mimic_path, 'mimic-cxr-2.0.0-split.csv'))
trainset_dicoms = mimic_split[mimic_split['split'].isin(['train', 'validate'])]['dicom_id'].values
mimic_metadata = mimic_metadata[mimic_metadata['dicom_id'].isin(trainset_dicoms)]
mimic_metadata = mimic_metadata[['dicom_id', 'subject_id', 'study_id']]

mimic_negbio = pd.read_csv(os.path.join(mimic_path, 'mimic-cxr-2.0.0-negbio.csv'))

df = pd.merge(mimic_negbio, mimic_metadata, how='inner', on=['subject_id', 'study_id'])

# # Count each category
# cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
#         'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
#         'Lung Opacity', 'No Finding', 'Pleural Effusion',
#         'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

# df.fillna(0).replace(-1, 0)[cols].sum()

new_entries = []
for i, row in tqdm(df.iterrows(), total=len(df)):    
    # new_entry = {k:v for k,v in row.to_dict().items() if k in ['subject_id', 'study_id', 'dicom_id']}
    new_entry = {}
    new_entry['dataset'] = 'mimic'
    img_path = os.path.join(mimic_path, 'files',
                            f"p{str(row['subject_id'])[:2]}",
                            f"p{str(row['subject_id'])}",
                            f"s{str(row['study_id'])}",
                            row['dicom_id']+'.jpg'
                            )
    new_entry['img_path'] = img_path
    
    if row['Atelectasis'] == 1:
        new_entry['atelectasis'] = 1
    else:
        new_entry['atelectasis'] = 0
    
    if row['Cardiomegaly'] == 1:
        new_entry['cardiomegaly'] = 1
    else:
        new_entry['cardiomegaly'] = 0
        
    if row['Consolidation']==1 or row['Pneumonia']==1:
        new_entry['consolidation'] = 1
    else:
        new_entry['consolidation'] = 0
    
    if row['Edema'] == 1:
        new_entry['edema'] = 1
    else:
        new_entry['edema'] = 0
    
    if row['Pleural Effusion'] == 1:
        new_entry['effusion'] = 1
    else:
        new_entry['effusion'] = 0
        
    if row['Enlarged Cardiomediastinum'] == 1:
        new_entry['enlarged_cardiomediastinum'] = 1
    else:
        new_entry['enlarged_cardiomediastinum'] = 0
    
    if row['Lung Lesion'] == 1:
        new_entry['nodule'] = 1
    else:
        new_entry['nodule'] = 0
        
    if row['Lung Opacity'] == 1:
        new_entry['opacity'] = 1
    else:
        new_entry['opacity'] = 0
        
    if row['Pleural Other'] == 1:
        new_entry['pleural_other'] = 1
    else:
        new_entry['pleural_other'] = 0
    
    if row['Pneumothorax'] == 1:
        new_entry['pneumothorax'] = 1
    else:
        new_entry['pneumothorax'] = 0
        
    new_entries.append(new_entry)
    
new_mimic_df = pd.DataFrame(new_entries)


#%% CheXpert to Multi-CXR
chexpert_path = '/home/wonjun/data/CheXpert-v1.0'
chexpert_metadata = pd.read_csv(os.path.join(chexpert_path, 'train.csv'))
chexpert_metadata = chexpert_metadata[chexpert_metadata['Frontal/Lateral']=='Frontal']

# cols = sorted(['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
#         'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
#         'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
#         'Pleural Other', 'Fracture', 'Support Devices'])
# chexpert_metadata[cols].fillna(0).replace(-1, 0).sum()

new_entries = []
for i, row in tqdm(chexpert_metadata.iterrows(), total=len(chexpert_metadata)):
    new_entry = {}
    new_entry['dataset'] = 'chexpert'
    
    img_path = os.path.join(os.path.dirname(chexpert_path), row['Path'])    
    new_entry['img_path'] = img_path
    
    if row['Atelectasis'] == 1:
        new_entry['atelectasis'] = 1
    else:
        new_entry['atelectasis'] = 0
    
    if row['Cardiomegaly'] == 1:
        new_entry['cardiomegaly'] = 1
    else:
        new_entry['cardiomegaly'] = 0
        
    if row['Consolidation']==1 or row['Pneumonia']==1:
        new_entry['consolidation'] = 1
    else:
        new_entry['consolidation'] = 0
    
    if row['Edema'] == 1:
        new_entry['edema'] = 1
    else:
        new_entry['edema'] = 0
    
    if row['Pleural Effusion'] == 1:
        new_entry['effusion'] = 1
    else:
        new_entry['effusion'] = 0
        
    if row['Enlarged Cardiomediastinum'] == 1:
        new_entry['enlarged_cardiomediastinum'] = 1
    else:
        new_entry['enlarged_cardiomediastinum'] = 0
    
    if row['Lung Lesion'] == 1:
        new_entry['nodule'] = 1
    else:
        new_entry['nodule'] = 0
        
    if row['Lung Opacity'] == 1:
        new_entry['opacity'] = 1
    else:
        new_entry['opacity'] = 0
        
    if row['Pleural Other'] == 1:
        new_entry['pleural_other'] = 1
    else:
        new_entry['pleural_other'] = 0
    
    if row['Pneumothorax'] == 1:
        new_entry['pneumothorax'] = 1
    else:
        new_entry['pneumothorax'] = 0
    
    new_entries.append(new_entry)

new_chexpert_df = pd.DataFrame(new_entries)
# new_chexpert_df[categories].sum()

# %% NIH-CXR to Multi-CXR
nih_path = '/home/wonjun/data/NIH-CXR'
nih_metadata = pd.read_csv(os.path.join(nih_path, 'Data_Entry_2017_with_paths.csv'), index_col=0)

nih_trainset_path = os.path.join(nih_path, 'train_val_list.txt')
with open(nih_trainset_path, 'r') as file:
    nih_trainset = file.readlines()
nih_trainset = [line.strip() for line in nih_trainset]
nih_metadata = nih_metadata[nih_metadata['Image Index'].isin(nih_trainset)]

# # Count
# # label_set = set()
# labels_count = {
#     'Atelectasis': 0,
#     'Cardiomegaly': 0,
#     'Consolidation': 0,
#     'Edema': 0,
#     'Effusion': 0,
#     'Emphysema': 0,
#     'Fibrosis': 0,
#     'Hernia': 0,
#     'Infiltration': 0,
#     'Mass': 0,
#     'No Finding': 0,
#     'Nodule': 0,
#     'Pleural_Thickening': 0,
#     'Pneumonia': 0,
#     'Pneumothorax': 0,
# }
# for i, row in tqdm(nih_metadata.iterrows(), total=len(nih_metadata)):    
#     labels = row['Finding Labels'].split("|")
#     # label_set.update(labels)
#     for label in labels:
#         labels_count[label] += 1
# labels_count

new_entries = []
for i, row in tqdm(nih_metadata.iterrows(), total=len(nih_metadata)):
    new_entry = {}
    new_entry['dataset'] = 'nih'
    
    img_path = os.path.join(nih_path, row['Image Paths'])
    new_entry['img_path'] = img_path
    
    labels = row['Finding Labels'].split("|")
    if 'Atelectasis' in labels:
        new_entry['atelectasis'] = 1
    else:
        new_entry['atelectasis'] = 0
    
    if 'Cardiomegaly' in labels:
        new_entry['cardiomegaly'] = 1
    else:
        new_entry['cardiomegaly'] = 0
    
    if ('Consolidation' in labels) or ('Pneumonia' in labels):
        new_entry['consolidation'] = 1
    else:
        new_entry['consolidation'] = 0
    
    if 'Edema' in labels:
        new_entry['edema'] = 1
    else:
        new_entry['edema'] = 0
        
    if 'Effusion' in labels:
        new_entry['effusion'] = 1
    else:
        new_entry['effusion'] = 0
        
    new_entry['enlarged_cardiomediastinum'] = 0 #-100
    
    if 'Nodule' in labels or 'Mass' in labels:
        new_entry['nodule'] = 1
    else:
        new_entry['nodule'] = 0
        
    new_entry['opacity'] = 0 #-100
    
    if 'Pleural_Thickening' in labels:
        new_entry['pleural_other'] = 1
    else:
        new_entry['pleural_other'] = 0
        
    if 'Pneumothorax' in labels:
        new_entry['pneumothorax'] = 1
    else:
        new_entry['pneumothorax'] = 0
    
    new_entries.append(new_entry)

new_nih_df = pd.DataFrame(new_entries)

    
#%% VinDr to Multi-CXR
vindr_path = '/home/wonjun/data/VinDr-CXR-jpgs'
vindr_metadata = pd.read_csv(os.path.join(vindr_path, 'image_labels_train.csv'))
# cols = sorted(['Aortic enlargement', 'Atelectasis',
#        'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation',
#        'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration',
#        'Lung Opacity', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',
#        'Nodule/Mass', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
#        'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD',
#        'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases',
#        'No finding']) 
# df[cols].sum()

new_entries = []
for i, row in tqdm(vindr_metadata.iterrows(), total=len(vindr_metadata)):
    new_entry = {}
    new_entry['dataset'] = 'vindr'
    
    img_path = os.path.join(vindr_path, 'train', row['image_id']+'.jpg')
    new_entry['img_path'] = img_path

    if row['Atelectasis'] == 1:
        new_entry['atelectasis'] = 1
    else:
        new_entry['atelectasis'] = 0
        
    if row['Cardiomegaly'] == 1:
        new_entry['cardiomegaly'] = 1
    else:
        new_entry['cardiomegaly'] = 0
        
    if row['Consolidation'] == 1 or row['Pneumonia'] == 1:
        new_entry['consolidation'] = 1
    else:
        new_entry['consolidation'] = 0    
        
    if row['Edema'] == 1:
        new_entry['edema'] = 1
    else:
        new_entry['edema'] = 0
        
    if row['Pleural effusion'] == 1:
        new_entry['effusion'] = 1
    else:
        new_entry['effusion'] = 0
        
    if row['Aortic enlargement'] == 1 or row['Enlarged PA'] == 1:
        new_entry['enlarged_cardiomediastinum'] = 1
    else:
        new_entry['enlarged_cardiomediastinum'] = 0
        
    if row['Lung tumor'] == 1 or row['Nodule/Mass'] == 1:
        new_entry['nodule'] = 1
    else:
        new_entry['nodule'] = 0
        
    if row['Lung Opacity'] == 1:
        new_entry['opacity'] = 1
    else:
        new_entry['opacity'] = 0
        
    if row['Pleural thickening'] == 1:
        new_entry['pleural_other'] = 1
    else:
        new_entry['pleural_other'] = 0
        
    if row['Pneumothorax'] == 1:
        new_entry['pneumothorax'] = 1
    else:
        new_entry['pneumothorax'] = 0
        
    new_entries.append(new_entry)

new_vindr_df = pd.DataFrame(new_entries)
# new_vindr_df[categories].sum()


#%% PadChest to Multi-CXR
# padchest_path = '/home/wonjun/data/PadChest'
# padchest_metadata = pd.read_csv(os.path.join(padchest_path, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'), index_col=0)

# label_set = set()
# failed_inds = []
# for i, row in tqdm(padchest_metadata.iterrows(), total=len(padchest_metadata)):
#     try:
#         labels = eval(row['Labels'])
#         labels = [l.strip() for l in labels]
#         label_set.update(labels)
#     except:
#         failed_inds.append(i)

# %% BraX to Multi-CXR
# brax_path = '/home/wonjun/data/BraX/physionet.org/files/brax/1.1.0'
# brax_metadata = pd.read_csv(os.path.join(brax_path, 'master_spreadsheet_update.csv'), index_col=0)
# %%
multicxr_df = pd.concat([new_snu_df,
                         new_mimic_df,
                         new_chexpert_df,
                         new_nih_df,
                         new_vindr_df], axis=0)
multicxr_df.to_csv('data_preparation/multicxr.csv')

#%%
df = pd.read_csv('data_preparation/multicxr.csv', index_col=0)
df = df.reset_index(drop=True)
df
#%%
for i, row in tqdm(df.iterrows(), total=len(df)):
    if not os.path.isfile(row['img_path']):
        break

#%%
# Convert the multi-label columns to a binary array using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(df[categories].values)

# Generate a unique label for each row from the binary array
strat_labels = binary_labels.dot(1 << np.arange(binary_labels.shape[-1] - 1, -1, -1))

# Initialize StratifiedKFold
n_splits = 3  # Number of folds, adjust as needed
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create the 'fold' column and assign fold numbers
df['fold'] = -1  # Initialize with -1 or another value indicating unassigned
for fold_number, (_, val_index) in enumerate(skf.split(X=df, y=strat_labels)):
    df.loc[val_index, 'fold'] = fold_number

#%%
df

#%%
df[df['fold']==4][categories].sum()

#%%
df = multicxr_df[multicxr_df['dataset'].isin(['snu', 'vindr'])][categories]
# %%
df[(df==0).all(axis=1)]
# %%
