#%%
import os#; os.environ['CUDA_VISIBLE_DEVICES']='2'
import json
import cv2
import random
import colorsys
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vit_o
import glob

seed = 2228
IND_TO_LABEL = {
    0: 'Atelectasis',
    1: 'Cardiomegaly',
    2: 'Consolidation',
    3: 'Edema',
    4: 'Enlarged Cardiomediastinum',
    5: 'Fracture',
    6: 'Lung Lesion',
    7: 'Lung Opacity',
    8: 'No Finding',
    9: 'Pleural Effusion',
    10: 'Pleural Other',
    11: 'Pneumonia',
    12: 'Pneumothorax',
    13: 'Support Devices',
}
LABEL_TO_IND = {v:k for k,v in IND_TO_LABEL.items()}


#%%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimic_root', type=str, default='/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512')
    parser.add_argument('--data_df_path', type=str, default='data_preparation/mimic_multilabel_all.csv')
    parser.add_argument('--ckpt_path', type=str, default='outputs/mimic-split-10-30-30-30/mimic_multilabel_fold2/checkpoint.pth')
    parser.add_argument('--label_of_interest', type=str,
                        choices=['Atelectasis',
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
                                 'Support Devices'],
                        default='Pleural Effusion')
    args = parser.parse_args("")
    return args

#%%
def load_img(img_path, img_size=(256,256), patch_size=8):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(img)
    img = pth_transforms.Compose(
        [
            utils.GaussianBlurInference(),
            pth_transforms.ToTensor()
        ]
    )(img) # ( 3, img_size[0], img_size[1] )
    
    # make the image divisible by patch size
    w, h = img.shape[1]-img.shape[1]%patch_size, img.shape[2]-img.shape[2]%patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    return img, w_featmap, h_featmap

def get_layer_selfattention(model: utils.MultiCropWrapper,
                             img: torch.Tensor,
                             layer_index: int,
                             w_featmap,
                             h_featmap,
                             patch_size=8):
    attentions = model.backbone.get_layer_selfattention(img, layer_index)
    attentions = attentions.detach()
    nh = attentions.shape[1]
    
    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=patch_size, mode='nearest'
    )[0].cpu().numpy()
    
    # im = np.transpose(img[0].detach().cpu().numpy(), (1,2,0))
    # fig, ax = plt.subplots(2,3, figsize=(9,6))
    # for j in range(nh):
    #     ax[j//3, j%3].imshow(im, alpha=0.5)
    #     ax[j//3, j%3].imshow(attentions[j], alpha=0.5)
    # fig.suptitle(f"attention maps for layer {layer_index}")
    # plt.show()
    
    return attentions

# def get_head_selfattention(model: utils.MultiCropWrapper,
#                            img: torch.Tensor,
#                            layer_index:int,
#                            head_index:int,
#                            patch_size=8):
#     attentions = model.backbone.get_layer_selfattention(img, layer_index)
#     attentions = attentions.detach()
#     nh = attentions.shape[1]
    
#     # keep only the output patch attention
#     attentions = attentions[0,:,0,1:].reshape(nh,-1)
#     attentions = attentions.reshape(nh, w_featmap, h_featmap)
#     attention = attentions[head_index]
#     attention = nn.functional.interpolate(
#         attention.unsqueeze(0).unsqueeze(0),
#         scale_factor=patch_size,
#         mode='nearest'
#     )[0][0].cpu().numpy()
    
#     im = np.transpose(img[0].detach().cpu().numpy(), (1,2,0))
#     fig, ax = plt.subplots(1,2, figsize=(8,4))
#     ax[0].imshow(im)
#     ax[1].imshow(im, alpha=0.5)
#     ax[1].imshow(attention, alpha=0.5)
#     fig.suptitle(f"layer {layer_index} head {head_index} attention map")
#     plt.show()
    
#     return attention

def get_head_importance(classes_of_interest: list,
                        model,
                        img):
    
    head_importance_by_class = {}
    # fig, ax = plt.subplots(1, len(classes_of_interest), figsize=(5*len(classes_of_interest), 5))
    if len(classes_of_interest)==1: ax=[ax]

    for i, _class in enumerate(classes_of_interest):
        class_ind = LABEL_TO_IND[_class]
        pred = model(img)
        pred[class_ind].backward()
        head_importance = []
        for block in model.backbone.blocks:
            ctx = block.attn.context_layer_val
            grad_ctx = ctx.grad
            dot = torch.einsum("bhli,bhli->bhl", [grad_ctx, ctx])
            # head_importance.append(dot.abs().sum(-1).sum(0).detach())
            head_importance.append(dot.sum(-1).sum(0).detach())
        head_importance = torch.vstack(head_importance)

        # Normalize attention values by layer
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        
        head_importance = head_importance.detach().cpu().numpy().astype(np.float32)
        # ax[i].imshow(head_importance, cmap='plasma')
        # ax[i].set_title(_class)
        
        head_importance_by_class[_class] = head_importance

        del pred
    
    plt.show()
    
    return head_importance_by_class



# %% 
def load_model(ckpt_path):
    # ckpt_path = 'outputs/mimic-split-10-30-30-30/mimic_multilabel_fold2/checkpoint.pth'
    CHECKPOINT_KEY = 'student'
    patch_size, out_dim, n_classes = 8, 65536, len(IND_TO_LABEL)
    model = vit_o.__dict__['vit_small'](patch_size=patch_size)
    embed_dim = model.embed_dim
    model = utils.MultiCropWrapper(
        model,
        vit_o.DINOHead(in_dim=embed_dim, out_dim=out_dim),
        vit_o.CLSHead(in_dim=384, hidden_dim=256, num_classes=n_classes)
    )

    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd[CHECKPOINT_KEY]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    msg = model.load_state_dict(sd)
    print(msg)
    model = model.to('cuda')
    return model

def save_attention_map(model, row, output_folder):
    """
    Save attention map for a single image
    
    row: pd.Series
        a row from one of the DataFrames in data_preparation folder

    """
    img_path = os.path.join(args.mimic_root, 'files',
                            f"p{str(int(row['subject_id']))[:2]}",
                            f"p{str(int(row['subject_id']))}",
                            f"s{str(int(row['study_id']))}",
                            row['dicom_id']+'.jpg')

    img, w_featmap, h_featmap = load_img(
        img_path=img_path, img_size=(256,256)
    )
    img = img.to('cuda')
    
    last_layer_attentions = get_layer_selfattention(model, img, w_featmap=w_featmap, h_featmap=h_featmap, layer_index=-1)
    
    pred = model(img)
    pred = np.array([t.detach().cpu().numpy()[0][0] for t in pred])
    predicted_classes = np.argsort(pred)[::-1] # list items in order of decreasing value
    predicted_classes = [ind for ind in predicted_classes if ind in np.where(pred>=0)[0]]
    predicted_classes = [IND_TO_LABEL[ind] for ind in predicted_classes]
    
    if len(predicted_classes) > 0:
        head_importance_by_class = get_head_importance(predicted_classes, model, img)
  
    best_head_ind_for_each_class = {}
    for k, v in head_importance_by_class.items():
        last_layer_best_head_ind = np.argmax(v[-1])
        best_head_ind_for_each_class[k] = int(last_layer_best_head_ind)
    
    save_dir = os.path.join(output_folder,
                             f"p{str(int(row['subject_id']))[:2]}",
                             f"p{str(int(row['subject_id']))}",
                             f"s{str(int(row['study_id']))}")
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, row['dicom_id']+'_best_head_indices.json')
    with open(json_path, 'w') as handle:
        json.dump(best_head_ind_for_each_class, handle, indent=2)
    attn_path = os.path.join(save_dir, row['dicom_id']+'.npy')
    np.save(attn_path, last_layer_attentions)    

#%% Save attention maps
def main(args):
    model = load_model(args.ckpt_path)
    
    # =========== Load an image ==========
    df = pd.read_csv(args.data_df_path)    
    df = df[df[args.label_of_interest]==1] # DataFrame with entries positive for label_of_interest
    
    ind = random.randint(0, len(df)) # pick a row at random
    row = df.iloc[ind]
    true_classes = list(row.index[row==1])
    img_path = os.path.join(args.mimic_root, 'files',
                            f"p{str(int(row['subject_id']))[:2]}",
                            f"p{str(int(row['subject_id']))}",
                            f"s{str(int(row['study_id']))}",
                            row['dicom_id']+'.jpg')

    img, w_featmap, h_featmap = load_img(
        img_path=img_path, img_size=(256,256)
    )
    img = img.to('cuda')
    print(row)
    print()

    # =========== Get model preds ============
    pred = model(img)
    pred = np.array([t.detach().cpu().numpy()[0][0] for t in pred])

    predicted_classes = np.argsort(pred)[::-1] # list items in order of decreasing value
    predicted_classes = [ind for ind in predicted_classes if ind in np.where(pred>=0)[0]]
    predicted_logits = [pred[i] for i in predicted_classes]
    predicted_classes = [IND_TO_LABEL[ind] for ind in predicted_classes]
    print("Predicted classes: ")
    print(predicted_classes)
    print(predicted_logits)
    print()
    
    print("True classes: ")
    print(true_classes)

    head_importance_by_class = get_head_importance(true_classes,
                                                   model,
                                                   img)

    attentions = get_layer_selfattention(model, img, w_featmap=w_featmap, h_featmap=h_featmap, layer_index=-1)
    
    im = np.transpose(img[0].detach().cpu().numpy(), (1,2,0))
    fig, ax = plt.subplots(2,3, figsize=(9,6))
    for j in range(6):
        ax[j//3, j%3].imshow(im, alpha=0.5)
        ax[j//3, j%3].imshow(attentions[j], alpha=0.5)
    # fig.suptitle(f"attention maps for layer {layer_index}")
    plt.show()


#%%
if __name__ == '__main__':
    args = parse_args()
    main(args)
# %%
