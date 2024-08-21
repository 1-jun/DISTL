#%%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
# %%
attention_maps_dir = '/media/wonjun/HDD8TB/DISTL-attention-maps/mimic-split-10-30-30-30'
subject_id = 10002013
study_id = 52163036
dicom_id = "1e647043-eed3576e-3123c170-780cb897-93a89502"
attn_path = os.path.join(attention_maps_dir,
                         f"p{str(int(subject_id))[:2]}",
                         f"p{str(int(subject_id))}",
                         f"s{str(int(study_id))}",
                         dicom_id+'.npy')
json_path = os.path.join(attention_maps_dir,
                         f"p{str(int(subject_id))[:2]}",
                         f"p{str(int(subject_id))}",
                         f"s{str(int(study_id))}",
                         dicom_id+'_best_head_indices.json')
attn_maps = np.load(attn_path)
with open(json_path, 'r') as handle:
    head_inds = json.load(handle)
# %%
fig, ax = plt.subplots(2,3, figsize=(9,6))
for i, img in enumerate(attn_maps):
    ax[i//3, i%3].imshow(img)
#%%
head_inds
