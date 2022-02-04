import os
from glob import glob
import json
import random

import cv2
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from tqdm import tqdm
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def fix(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.daterministic=True
  torch.backends.cudnn.benchmark=False
  np.random.seed(seed)
  random.seed(seed)

fix(42)

path = './'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_model_path = './cutmix_statedict.pth'

def split(train_label):
  kfold = StratifiedKFold(n_splits=10)
  train_label['fold'] = -1

  for idx, (tr_idx, val_idx) in enumerate(kfold.split(train_label.image.values, train_label.label.values)):
    train_label.loc[val_idx, 'fold'] = idx

  val_idx = list(train_label[train_label.fold == 9].index) + list(train_label[train_label.fold == 10].index)
  train_idx = set(train_label.index) - set(val_idx)

  train_idx = sorted(list(train_idx))
  val_idx = sorted(list(val_idx))

  train_df = train_label.loc[train_idx].reset_index(drop=True)
  val_df = train_label.loc[val_idx].reset_index(drop=True)

  return train_df ,val_df

train_transform = A.Compose([
  A.Resize(256, 256, interpolation=cv2.INTER_AREA, always_apply=True),
  A.HorizontalFlip(p=0.5),
  A.VerticalFlip(p=0.5),
  A.Rotate(p=0.5),
  A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
  ToTensor()
])

valid_transform = A.Compose([
  A.Resize(256, 256, interpolation=cv2.INTER_AREA, always_apply=True),
  A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
  ToTensor()
])

class LGDataset(Dataset):
  def __init__(self, file_df, transform):
    self.file_df = file_df
    self.path = path
    self.transform = transform
    self.label_to_idx = label_to_idx

  def __len__(self):
    return self.file_df.shape[0]

  def __getitem__(self, idx):
    image_name = int(self.file_df.iloc[idx, 0])
    img_path = os.path.join(self.path, f'train/{image_name}/{image_name}.jpg')
    label = str(self.file_df.iloc[idx, 1])

    img = cv2.imread(img_path)
    if self.transform is not None:
      img = self.transform(image = img)['image']

    label = torch.tensor(self.label_to_idx[label], dtype=torch.long)

    return {
      'img': img,
      'label': label
    }

class CNN(nn.Module):
  def __init__(self, name:str, class_n):
    super(CNN,self).__init__()
    self.model = EfficientNet.from_pretrained(name, num_classes=class_n)

  def forward(self, x):
    return self.model(x)

## CutMix
def rand_bbox(size, lam):
  W = size[2]
  H = size[3]
  cut_rat = np.sqrt(1. - lam)
  cut_w = np.int(W * cut_rat)
  cut_h = np.int(H * cut_rat)

  cx = np.random.randint(W)
  cy = np.random.randint(H)

  bbx1 = np.clip(cx - cut_w // 2, 0 ,W)
  bby1 = np.clip(cy - cut_h // 2, 0, H)
  bbx2 = np.clip(cx + cut_w // 2 ,0, W)
  bby2 = np.clip(cy + cut_h // 2, 0, H)

  return bbx1, bby1, bbx2, bby2

## CutMix Training
def Train(model, epochs):

  print("Start Training")
  best_score = -1
  model = model.to(device)

  for epoch in tqdm_notebook(range(epochs)):
    model.train()
    train_loss = []
    for data in train_loader:
      optimizer.zero_grad()
      img = data['img'].to(device)
      label = data['label'].to(device)

      if np.random.rand() < 0.5:
        cost = model(img)
        loss = loss_fn(cost, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

      else:
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(img.size()[0])

        shuffled_label = label[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))

        cost = model(img)
        loss = loss_fn(cost, label) * lam + loss_fn(cost, shuffled_label) * (1 - lam)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    val_loss = []
    val_true = []
    val_pred = []

    model.eval()
    with torch.no_grad():
      for data in val_loader:
        img = data['img'].to(device)
        label = data['label'].to(device)

        cost = model(img)
        loss = loss_fn(cost, label)
        val_loss.append(loss.item())
        val_true.extend(label.cpu().numpy())
        val_pred.extend(torch.argmax(cost, dim=1).cpu().numpy())

    lr_sc.step(np.mean(val_loss))
    score = f1_score(val_true, val_pred, average='macro')

    if score > best_score:
      best_score = score
      torch.save(model.state_dict(), best_model_path)

    print(
      f"epoch:{epoch + 1}, train_loss : {np.mean(train_loss):.5f}, val_loss : {np.mean(val_loss):.5f}, f1_score: {score:.5f}")

  print("End Training")


if __name__ == "__main__":

  train_label = pd.read_csv('./train.csv')
  label_kind = np.unique(train_label.label.values).tolist()
  label_to_idx = {i: idx for idx, i in enumerate(label_kind)}
  idx_to_label = {idx: i for idx, i in enumerate(label_kind)}

  train_df , val_df = split(train_label)

  ## Dataset, DataLoader
  train_dataset = LGDataset(file_df=train_df, transform=train_transform)
  val_dataset = LGDataset(file_df=val_df, transform=valid_transform)

  train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=2)
  val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=2)

  ## Model
  model = CNN(name='efficientnet-b2', class_n=len(label_to_idx))

  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  lr_sc = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, verbose=True)

  ## Training
  Train(model = model ,epochs = 30)

