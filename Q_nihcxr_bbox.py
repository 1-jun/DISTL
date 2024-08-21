#%%
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from IPython.display import display

# %%
nih_root = '/media/wonjun/HDD2TB/NIH-CXR'
bboxes_path = os.path.join(nih_root, 'BBox_List_2017.csv')
bboxes = pd.read_csv(bboxes_path)
bboxes
#%%
metadata_path = os.path.join(nih_root, 'Data_Entry_2017_with_paths.csv')
metadata = pd.read_csv(metadata_path)
metadata

#%%
df = pd.merge(bboxes, metadata, how='left', on=['Image Index'])
df = df[df['Finding Label'] == 'Nodule']
df
#%%
i = 29
row = df.iloc[i]
display(row)

image_path = os.path.join(nih_root, row['Image Paths'])
image = Image.open(image_path)
print(image.size)
image = np.array(image)
fig, ax = plt.subplots(1,1)
ax.imshow(image, cmap='gray')

x, y = row['Bbox [x'], row['y']
w, h = row['w'], row['h]']
bbox = Rectangle((x, y), w, h, edgecolor='red', facecolor='none')
ax.add_patch(bbox)

ax.set_title(row['Finding Label'])

#%%