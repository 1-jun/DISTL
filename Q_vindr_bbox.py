#%%
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %%
vindr_root = '/media/wonjun/HDD2TB/VinDr-CXR-jpgs'
annotations_path = os.path.join(vindr_root, 'annotations_test.csv')
annots = pd.read_csv(annotations_path)
#%%
df = annots[annots['class_name']=='Nodule/Mass']
df
# %%
i = 43
row = df.iloc[i]
print(row)

image_path = os.path.join(vindr_root, 'test', row['image_id']+'.jpg')
image = Image.open(image_path)
print(image.size)
image = np.array(image)
fig, ax = plt.subplots(1,1)
ax.imshow(image, cmap='gray')

w = row['x_max'] - row['x_min']
h = row['y_max'] - row['y_min']
bbox = Rectangle((row['x_min'], row['y_min']), w, h,
                 edgecolor='red', facecolor='none')
ax.add_patch(bbox)

#%%