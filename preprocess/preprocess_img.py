### resize the original image and save as tensors

import json
import os
import torchvision.transforms as transforms
import torch
from PIL import Image

data_path="/home/guests/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/files/"
label_path="/home/guests/mlmi_kamilia/RadGraph Relationformer_matcher1/datasets/radgraph/"
out = '/home/guests/data/DIVA/mimic/mimic-cxr-jpg-resized/'

#os.system(command)
with open(label_path + "train_all.json", 'r') as f:  # dev_matcher1.json
    data = json.load(f)

for key in data:
    [group_id, patient_id, study_id] = key.split('/')
    [study_id, _] = study_id.split('.')
    img_path = data_path + group_id + '/' + patient_id + '/' + study_id + '/'

    for file in os.listdir(img_path):
        if file.endswith('.jpg'):
            img_path = (os.path.join(img_path, file))
            out_path = out + group_id + '/' + patient_id + '/' + study_id + '/'
            break

    ### transform

    image = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0)),
        transforms.Resize([256, 256]),
        transforms.Normalize([0], [255])
    ])

    image_tensor = transform(image)

    ### save the image as tensors
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(image_tensor,out_path+"image.pt")

    break

print("finished")
