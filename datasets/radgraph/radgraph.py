import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch

import json

from torch.utils.data import Dataset

MAX_TOKENS = 45
MAX_EDGES = 117

class Radgraph(Dataset):
    def __init__(self, is_train, is_augment,
                 data_path="/home/data/DIVA/mimic/mimic-cxr-jpg-resized/",
                 label_path="/home/guests/mlmi_kamilia/Radgraphformer_emb/datasets/radgraph/"):
        # is_train: training set or val set
        # is_augment: augment or not
        super(Radgraph, self).__init__()

        self.is_train = is_train
        self.data_path = data_path
        self.label_path = label_path
        self.is_augment = is_augment
        self.data = None

        if self.is_train:
            with open(self.label_path + "train_all.json", 'r') as f:
                self.data = json.load(f)

        else:
            with open(self.label_path + "dev_all.json", 'r') as f:
                self.data = json.load(f)

        self.idx = list(self.data.keys())

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        report_id = self.idx[idx]
        # group_id: p10, patient_id:p18004941, study_id = s588...
        [group_id, patient_id, study_id] = report_id.split('/')
        [study_id, _] = study_id.split('.')
        self.img_path = self.data_path + group_id + '/' + patient_id + '/' + study_id

        img_path = self.img_path + '/image.pt'
        if os.path.isfile(img_path):
            img_tensor = torch.load(img_path)
        else:
            raise Exception("Sorry, no img found at specified file")

        report_ls = self.data[report_id]  # list(N,3)
        labels_ls = []
        tokens_ls = []
        edges_ls = []  # report_ls['edges']
        for i in range(MAX_TOKENS):
            if i < len(report_ls['labels']):
                # token
                tokens_ls.append(report_ls['tokens'][i] + 1)  # +1 to make none class as 0
                # label
                if report_ls['labels'][i] == "ANAT-DP":
                    labels_ls.append(1)
                elif report_ls['labels'][i] == 'OBS-DP':
                    labels_ls.append(2)
                elif report_ls['labels'][i] == 'OBS-U':
                    labels_ls.append(3)
                else:
                    labels_ls.append(0)
            else:
                labels_ls.append(0)  # invalid objects' label
                tokens_ls.append(0)  # invalid tokens' class
        for i in range(MAX_EDGES):
            if i < len(report_ls['edges']):
                edges_ls.append([report_ls['edges'][i][0], report_ls['edges'][i][1],
                                 report_ls['edges'][i][2] + 1])  # +1 makes invalid relation label as 0
            else:
                edges_ls.append([MAX_TOKENS, MAX_TOKENS, 0])  # invalid relation

        rect = {'tokens': torch.LongTensor(tokens_ls), 'labels': torch.LongTensor(labels_ls),
                'edges': torch.LongTensor(edges_ls), 'imgs_ls': img_tensor.repeat(3, 1, 1), 'id': report_id}
        return rect


if __name__ == "__main__":
    Data = Radgraph(is_train=True, is_augment=False,
                    data_path="/home/guests/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/files/",
                    label_path="/home/yiheng/Desktop/RadGraph Relationformer/datasets/radgraph/")
