#%%
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
img_name = '051b7911-cb00aec9-0b309188-89803662-303ec278'
npys_dir = 'attention_maps_progression/mimic_multilabel_useall_mainrun'
save_dir = 'attention_maps_progression/mimic_multilabel_useall_mainrun_jpgs'
#%%
fnames = os.listdir(npys_dir)
fnames = sorted([file for file in fnames if img_name in file], key=lambda x: (len(x), x))
# arrs = [np.load(os.path.join(npys_dir, npy)) for npy in fnames]
# %%
for j, fname in enumerate(tqdm(fnames)):
    
    arr = np.load(os.path.join(npys_dir, fname))
    
    fig, axs = plt.subplots(2,3, figsize=(9,6))
    axs = axs.flatten()

    for i in range(6):
        axs[i].imshow(arr[i])
        axs[i].axis('off')
    plt.suptitle(fname[:-4])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname[:-4]+'.jpg'),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()
# %%