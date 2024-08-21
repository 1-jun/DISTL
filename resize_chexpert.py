#%%
import os
import cv2
import glob
import shutil
import zipfile
from tqdm import tqdm
from PIL import Image
from pathlib import Path

#%%
root = '/media/wonjun/HDD2TB/CheXpert-v1.0/train'
img_paths = glob.glob(os.path.join(root, '**/*.jpg'), recursive=True)
target_dir = '/media/wonjun/HDD2TB/chexpert-resized512/train'

#%%
for img_path in tqdm(img_paths):
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
    target_path = os.path.join(target_dir, *str(img_path).split('/')[-3:])
    target_path_parent = Path(target_path).parent
    os.makedirs(target_path_parent, exist_ok=True)
    img_resized.save(target_path)
#%%

#%%


#%%
for folder in img_folders:
    

# %%
source_dir = img_folders[0]
source_dir.split('/')[-1]
#%%
for source_dir in img_folders:
    folder_num = source_dir.split('/')[-1]
    print(folder_num)
    target_dir = f'/media/wonjun/TOSHIBA8TB/padchest-resizedt512/{folder_num}'

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over all files in the source directory
    for filename in tqdm(os.listdir(source_dir)):
        if filename.endswith('.png'):
            # Open the image
            img_path = os.path.join(source_dir, filename)
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
            target_path = os.path.join(target_dir, filename)
            img_resized.save(target_path)


# %%
