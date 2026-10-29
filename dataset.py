import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class BimaOCRDataset(Dataset):
    def __init__(self, csv_file, img_dir, char2idx, img_h=32, max_w=256):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.char2idx = char2idx
        self.img_h = img_h
        self.max_w = max_w

    def __len__(self):
        return len(self.df)

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((self.img_h, self.max_w), 255, dtype=np.uint8)
            
        h, w = img.shape
        new_w = min(self.max_w, max(16, int(round(w * self.img_h / h))))
        resized = cv2.resize(img, (new_w, self.img_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.full((self.img_h, self.max_w), 255, dtype=np.uint8)
        canvas[:, :new_w] = resized
        
        tensor_img = canvas.astype(np.float32) / 255.0
        tensor_img = (tensor_img - 0.5) / 0.5
        tensor_img = np.expand_dims(tensor_img, axis=0)
        return torch.tensor(tensor_img, dtype=torch.float32), new_w

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['filename']))
        image_tensor, original_width = self.preprocess_image(img_path)
        
        label_text = str(row['label']).lower()
        target = [self.char2idx[c] for c in label_text if c in self.char2idx]
        return image_tensor, torch.tensor(target, dtype=torch.long), original_width, label_text

def ctc_collate_fn(batch):
    images, targets, widths, texts = zip(*batch)
    images = torch.stack(images, 0)
    
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets)
    input_lengths = torch.tensor([max(1, w // 4) for w in widths], dtype=torch.long)
    
    return images, flat_targets, input_lengths, target_lengths, texts
