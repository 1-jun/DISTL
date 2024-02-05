### Lesion Localization with Attention Maps
```
> python Q_localization.py --mimic_root /PATH/TO/MIMIC/DIRECTORY/ --ckpt_path /PATH/TO/PRETRAINED/CHECKPOINT/ --label_of_interest "Pleural Effusion"
```
`label_of_interest` can be one of ['Atelectasis',
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
<br><br>
This will generate two images:
* heat map that shows importance of each attention head in making the prediction for that label
* attention maps of the last layer attention heads

Note: head importance does not work well currently.. Need better way to select attention head to choose which attention map to use.