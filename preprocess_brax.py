#%%
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# %% Split into Train/Val (the split df's are already saved)
# brax_root = '/media/wonjun/HDD8TB/brax'
# df = pd.read_csv(os.path.join(brax_root, 'master_spreadsheet_update.csv'), index_col=0)
# unique_patient_ids = df['PatientID'].unique()
# train_ids, val_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)
# train_df = df[df['PatientID'].isin(train_ids)]
# val_df = df[df['PatientID'].isin(val_ids)]
# train_df.to_csv(os.path.join(brax_root, 'Q_train_split.csv'))
# val_df.to_csv(os.path.join(brax_root, 'Q_val_split.csv'))

#%%
# brax_root = '/media/wonjun/HDD8TB/brax'
# df = pd.read_csv(os.path.join(brax_root, 'Q_train_split.csv'), index_col=0)
# df = df[df['ViewPosition'].isin(['PA', 'AP'])]

#%%
src_dir = '/media/wonjun/HDD8TB/brax/images'
pngs_list = glob.glob(os.path.join(src_dir, '**/*.png'), recursive=True)
pngs_list
#%%
failed = []
target_dir = '/media/wonjun/HDD8TB/brax-resized512/images'
for i, img_path in tqdm(enumerate(pngs_list), total=len(pngs_list)):    
    target_path = os.path.join(target_dir, *img_path.split('/')[-4:])
    target_path_parent = Path(target_path).parent
    os.makedirs(target_path_parent, exist_ok=True)
    if os.path.isfile(target_path):
        continue

    try:
        # Open the image
        img = Image.open(img_path)

        # Calculate the new size maintaining the aspect ratio
        width, height = img.size
        if width < height:
            new_width = 512
            new_height = int((512 / width) * height)
        else:
            new_height = 512
            new_width = int((512 / height) * width)

        # Resize the image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save the resized image to the target directory
        img_resized.save(target_path)
    except:
        print(img_path)
        failed.append(img_path)
#%%
"""These images are faulty! Skip them"""
p = '/media/wonjun/HDD8TB/brax/images/id_a8319cca-34351e63-024eb674-e34030b6-80bb3f40/Study_64491075.55988256.27027861.57546766.84854567/Series_12173925.70934735.79756216.27080791.76608566/image-22690942-78813103-68555148-48389446-07160114.png'
p = '/media/wonjun/HDD8TB/brax/images/id_1393d9d5-9f84b019-9687522a-1d2aaf1f-3959784e/Study_91606409.15658685.20833816.36407638.98330235/Series_71962691.91277396.32502660.73085698.66591371/image-37997198-18966152-20534320-02647727-39479040.png'

#%%
dst_dir = '/media/wonjun/HDD8TB/brax-resized512/images'

#%%
df[df['Lung Lesion']==1].
#%%
categories = ['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
       'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']
df[sorted(categories)].fillna(0).replace(-1, 1).sum()

# %%
row = df.iloc[23]
pngpath = os.path.join(brax_root, row['PngPath'])
img = Image.open(pngpath)
img
#%%
df['ViewPosition'].value_counts()
#%%
df['ViewPosition'].isna().sum()
#%%
row = df[df['Lung Lesion']==1].iloc[30]
pngpath = os.path.join(brax_root, row['PngPath'])
img = Image.open(pngpath)
img
# %%
row
#%%
row = df[df['PatientID']=='id_c43338e9-58b660b1-41e454a5-f67df48c-ca3596ea'].iloc[3]
pngpath = os.path.join(brax_root, row['PngPath'])
img = Image.open(pngpath)
img
#%%
df[df['PatientID']=='id_c43338e9-58b660b1-41e454a5-f67df48c-ca3596ea']