#%%
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

seed=2228

lesions = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    # 'Enlarged Cardiomediastinum',
    # 'Fracture',
    # 'Lung Lesion',
    # 'Lung Opacity',
    'Pleural Effusion',
    # 'Pleural Other',
    # 'Pneumonia',
    # 'Pneumothorax',
    # 'Support Devices'
]

def combine_cnsl_pna(row):
    """
    Combine columns 'Consolidation' and 'Pneumonia' into a single column
    """
    if row['Consolidation']==1 or row['Pneumonia']==1:
        return 1.0
    elif row['Consolidation']==-1 or row['Pneumonia']==-1:
        return -1.0
    else:
        return 0.0

# %%
mimic_root = '/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512'

metadata_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-metadata.csv')
mimic_split_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-split.csv')
negbio_labels_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-negbio.csv')
metadata = pd.read_csv(metadata_path)
mimic_split = pd.read_csv(mimic_split_path)
negbio_labels = pd.read_csv(negbio_labels_path)
negbio_labels = negbio_labels.fillna(0)
# negbio_labels = negbio_labels.replace(-1, 0)
negbio_labels = negbio_labels.drop('No Finding', axis=1)

mimic_trainset_df = mimic_split[mimic_split['split']=='train']
df = pd.merge(mimic_trainset_df, metadata[['dicom_id', 'ViewPosition', 'StudyDate', 'StudyTime']],
              on='dicom_id', how='inner')
df = df[df['ViewPosition'].isin(['PA', 'AP'])]


df = pd.merge(df,
              negbio_labels[['subject_id',
                             'study_id',
                             'Atelectasis',
                             'Cardiomegaly',
                             'Consolidation',
                             'Edema',
                             'Pleural Effusion',
                             'Pneumonia']],
              on=['subject_id', 'study_id'], how='left')
df['Cnsl_Pna'] = df.apply(combine_cnsl_pna, axis=1)
df = df.drop('Pneumonia', axis=1)
df = df.drop('Consolidation', axis=1)
df = df.rename(columns={'Cnsl_Pna':'Consolidation'})

#%%
dfs_for_label_training = []
for lesion in lesions:
    others = [l for l in lesions if l != lesion]
    
    # Collect the rows that are positive for one lesion but negative for all others
    only_one_lesion_df = df[
        (df[lesion] == 1) &
        (df[others] == 0).all(axis=1)
    ]
    
    n_positive_for_lesion = df[lesion].value_counts()[1.0]
    n_rows_for_label_training = int(n_positive_for_lesion * 0.1)
    
    # n_positive_only_for_lesion = len(only_one_lesion_df)
    # print(f"study_ids with {lesion}: {n_positive_for_lesion}")    
    # print(f"study_ids with only {lesion}: {n_positive_only_for_lesion}")
    # print(f"ratio: {n_positive_only_for_lesion/n_positive_for_lesion:.3f}")
    # print()
    
    _df = only_one_lesion_df.sample(n_rows_for_label_training)
    dfs_for_label_training.append(_df)

labeled_training_df = pd.concat(dfs_for_label_training)
labeled_training_df = labeled_training_df.sort_values(['subject_id', 'study_id'])
 

#%%
distl_training_df = df[~df['dicom_id'].isin(labeled_training_df['dicom_id'].values)]
distl_training_df = distl_training_df.fillna(0)
# distl_training_df = distl_training_df.replace(-1, 0)
distl_training_df[lesions] = distl_training_df[lesions].astype(int)
distl_training_df['for_split'] = (distl_training_df['Atelectasis'].replace(-1,0).astype(str)
                                  + distl_training_df['Cardiomegaly'].replace(-1,0).astype(str)
                                  + distl_training_df['Consolidation'].replace(-1,0).astype(str)
                                  + distl_training_df['Edema'].replace(-1,0).astype(str)
                                  + distl_training_df['Pleural Effusion'].replace(-1,0).astype(str))


#%%
# Split the DISTL set into three folds
folds = {}
kfold_split = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
for i, (_, fold_indices) in enumerate(kfold_split.split(distl_training_df, distl_training_df['for_split'])):
    folds[f"fold_{i}"] = distl_training_df.iloc[fold_indices]

#%%
# save
labeled_training_df.to_csv(os.path.join(f"mimic_5labels_labeled.csv"))
for fold_name, fold_df in folds.items():
    fold_df.to_csv(os.path.join(f"mimic_5labels_{fold_name}.csv"))