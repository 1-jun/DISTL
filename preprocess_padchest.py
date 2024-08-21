#%%
import os
import cv2
import shutil
import zipfile
from tqdm import tqdm
from PIL import Image, ImageFile

#%%
root = '/media/wonjun/TOSHIBA8TB/PadChest'

img_folders = sorted([f for f in os.listdir(root) if f.isnumeric()], key=lambda x:(len(x), x))
img_folders = [os.path.join(root, f) for f in img_folders]

#%%
failed = []
for source_dir in img_folders:
    folder_num = source_dir.split('/')[-1]
    print(folder_num)
    target_dir = f'/media/wonjun/TOSHIBA8TB/padchest-resizedt512/{folder_num}'

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over all files in the source directory
    for filename in tqdm(os.listdir(source_dir)):
        if filename.endswith('.png'):
            target_path = os.path.join(target_dir, filename)
            if os.path.isfile(target_path):
                continue
            
            try:
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
                img_resized.save(target_path)
            except:
                print(filename)
                failed.append(filename)
                
#%%
img_folders
#%%
failed_paths = [
 '13/216840111366964013686042548532013208193054515_02-026-007.png',
 '13/216840111366964013590140476722013058110301622_02-056-111.png',
 '16/216840111366964013649110343042013092101343018_02-075-146.png',
 '17/216840111366964013590140476722013049100117076_02-063-097.png',
 '17/216840111366964013590140476722013043111952381_02-065-198.png',
 '18/216840111366964013590140476722013028161046120_02-015-149.png',
 '19/216840111366964013829543166512013353113303615_02-092-190.png',
 '20/216840111366964013962490064942014134093945580_01-178-104.png',
 '41/216840111366964012373310883942009118095358945_00-070-125.png',
 '41/216840111366964013076187734852011209100952356_00-117-049.png',
 '41/216840111366964012989926673512011132200139442_00-157-099.png',
 '41/216840111366964012959786098432011047133018895_00-179-047.png',
 '41/216840111366964012989926673512011151082430686_00-157-045.png',
 '41/216840111366964012373310883942009190102257057_00-027-134.png',
 '42/216840111366964013076187734852011291090445391_00-196-188.png',
 '42/216840111366964012373310883942009117084022290_00-064-025.png',
 '43/216840111366964012339356563862009072111404053_00-043-192.png',
 '43/216840111366964012558082906712009301143450268_00-075-157.png',
 '43/216840111366964012283393834152009033102258826_00-059-087.png',
 '43/216840111366964012487858717522009280135853083_00-075-001.png',
 '43/216840111366964012989926673512011101154138555_00-191-086.png',
 '43/216840111366964012283393834152009033140208626_00-059-118.png',
 '44/216840111366964012373310883942009170084120009_00-097-074.png',
 '44/216840111366964012819207061112010281134410801_00-129-131.png',
 '44/216840111366964012373310883942009180082307973_00-097-011.png',
 '45/216840111366964012558082906712009300162151055_00-078-079.png',
 '45/216840111366964012339356563862009068084200743_00-045-105.png',
 '46/216840111366964012989926673512011074122523403_00-163-058.png',
 '46/216840111366964012373310883942009152114636712_00-102-045.png',
 '46/216840111366964012558082906712009327122220177_00-102-064.png',
 '46/216840111366964012989926673512011083134050913_00-168-009.png',
 '47/216840111366964013076187734852011178154626671_00-145-086.png',
 '47/216840111366964012959786098432011033083840143_00-176-115.png',
 '48/216840111366964012819207061112010306085429121_04-020-102.png',
 '48/216840111366964013076187734852011287092959219_00-195-171.png',
 '49/216840111366964012819207061112010307142602253_04-014-084.png',
 '49/216840111366964012819207061112010315104455352_04-024-184.png'
]
failed_paths = [os.path.join(root, p) for p in failed_paths]
failed_paths
#%%
target_dir = '/media/wonjun/TOSHIBA8TB/padchest-resized512'
for i, p in enumerate(failed_paths):
    target_path = os.path.join(target_dir, *p.split('/')[-2:])
    try:
        img = Image.open(p)
        width, height = img.size
        if width < height:
            new_width = 512
            new_height = int((512 / width) * height)
        else:
            new_height = 512
            new_width = int((512 / height) * width)

        # Resize the image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_resized.save(target_path)
    except Exception as e:
        print(i, e)

#%%
