#%%
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

#%%
def show_jpgs(
    subject_id = None,
    study_id = None,
    mimic_images_path = '/home/wonjun/data/mimic-cxr-jpg-resized512/files'
):
        
    images_path = os.path.join(mimic_images_path,
                            f'p{subject_id}'[:3],
                            f'p{subject_id}',
                            f's{study_id}')
    images = os.listdir(images_path)
    
    if len(images) > 1:
        fig, ax = plt.subplots(1, len(images))
        for i, _img_path in enumerate(images):
            img_path = os.path.join(images_path, _img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i].imshow(img)
            print(_img_path)
        plt.show()
    elif len(images) == 1:
        fig, ax = plt.subplots()
        img_path = os.path.join(images_path, images[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        print(img_path)
        plt.show()
    else:
        print("zero images in that path")
        
def show_report(
    subject_id = None,
    study_id = None,
    mimic_reports_path = '/home/wonjun/data/mimic-cxr-jpg-resized512/reports/files'
):
    report_path = os.path.join(mimic_reports_path,
                                f'p{subject_id}'[:3],
                                f'p{subject_id}',
                                f's{study_id}.txt')    
    with open(report_path, 'r') as report:
        txt = report.read()
    print(txt)

def show_jpg_for_dicom_id(dicom_id,
                          mimic_metadata_df,
                          mimic_images_path='/home/wonjun/data/mimic-cxr-jpg-resized512/files'):
    row = mimic_metadata_df[mimic_metadata_df['dicom_id']==dicom_id]
    img_path = os.path.join(mimic_images_path,
                            f"p{str(int(row['subject_id']))[:2]}",
                            f"p{str(int(row['subject_id']))}",
                            f"s{str(int(row['study_id']))}",
                            dicom_id+'.jpg')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1,1)
    ax.imshow(img)
    plt.show()
#%%
mimic_root = '/home/wonjun/data/mimic-cxr-jpg-resized512'
metadata_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-metadata.csv')
metadata = pd.read_csv(metadata_path)
#%%
dcm = '051b7911-cb00aec9-0b309188-89803662-303ec278'
show_jpg_for_dicom_id(dcm, metadata)

#%%
mimic_root = '/home/wonjun/data/mimic-cxr-jpg-resized512'
negbio_labels_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-negbio.csv')
df= pd.read_csv(negbio_labels_path)
df
#%%
df.columns
#%%
lesion = 'Pneumonia'
ind = 0

row = df[df[lesion]==1].iloc[ind]

subject_id = int(row.subject_id)
study_id = int(row.study_id)
show_report(subject_id, study_id)
show_jpgs(subject_id, study_id)

# %%
subject_id = 10002428
study_id = 59659695
show_report(subject_id, study_id)
show_jpgs(subject_id, study_id)