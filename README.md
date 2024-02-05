Localize lesions with attention visualization
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