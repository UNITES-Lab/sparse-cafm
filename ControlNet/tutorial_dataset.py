import json
import cv2
import numpy as np

from torch.utils.data import Dataset

TRAIN_PROP = 0.9
BOOTSTRAP_FACTOR = 1
DATASET_NAME = "bootstrapped-dataset-64x64"

class MyDataset(Dataset):
    def __init__(self, data):
        # HACK: for bootstrapping dataset
        self.data = data * BOOTSTRAP_FACTOR

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(f'./training/{DATASET_NAME}/' + source_filename)
        target = cv2.imread(f'./training/{DATASET_NAME}/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
    @staticmethod
    def get_train_val_datasets():
        data = []
        with open(f'./training/{DATASET_NAME}/prompt.json', 'rt') as f:
            for line in f:
                data.append(json.loads(line))
        # randomly shuffle data
        np.random.shuffle(data)
        # split data into 90/10 train/val splits
        return MyDataset(data[:int(len(data) * TRAIN_PROP)]), MyDataset(data[int(len(data) * TRAIN_PROP):])