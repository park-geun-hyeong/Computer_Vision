'''
Dacon LG AI Contest
https://dacon.io/competitions/official/235870/overview/description

image, sequence input, Multi label classification

'''

import os
from glob import glob
import json
import random
from collections import OrderedDict

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

## Label
train_label = pd.read_csv('./train.csv')
label_kind = np.unique(train_label.label.values).tolist()
label_to_idx = {i: idx for idx, i in enumerate(label_kind)}
idx_to_label = {idx: i for idx, i in enumerate(label_kind)}

## Dict for MinMaxScaling of sequence data
with open('./feature_min_max.json', 'r') as f:
  csv_feature_dict = json.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
max_len = 196
emb_dim = 256
num_features = len(csv_feature_dict)
class_n = len(label_to_idx)
rate = 0.1
batch_size = 16
epochs = 20
best_model_path = './best_model.pth'
ck_dir = './cutmix_statedict.pth'
path = './'
cnn_lr = 0.0001
lr = 0.001

def fix(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.daterministic=True
  torch.backends.cudnn.benchmark=False
  np.random.seed(seed)
  random.seed(seed)

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

# path = albsolute path for this project folder

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

TTA = A.Compose([
                             A.Resize(256,256, interpolation = cv2.INTER_AREA,always_apply=True),
                             A.OneOf([
                                      A.HorizontalFlip(p=0.5),
                                      A.VerticalFlip(p=0.5),
                                      A.Rotate(p=0.5),
                             ], p = 0.5),
                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                             ToTensor()
])

class LGDataset(Dataset):
  def __init__(self, file_df, transform=None):
    self.file_df = file_df
    self.path = path
    self.transform = transform
    self.csv_feature_dict = csv_feature_dict
    self.max_len = 196
    self.label_to_idx = label_to_idx

  def __len__(self):
    return self.file_df.shape[0]

  def __getitem__(self, idx):
    image_name = int(self.file_df.iloc[idx, 0])
    img_path = os.path.join(self.path, f'train/{image_name}/{image_name}.jpg')
    label = str(self.file_df.iloc[idx, 1])

    csv_path = os.path.join(self.path, f"train/{image_name}/{image_name}.csv")
    df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
    df = df.replace('-', np.nan)
    df.fillna(method='pad')

    for col in self.csv_feature_dict.keys():
      df[col] = df[col].astype(float) - self.csv_feature_dict[col]['MIN']
      df[col][df[col] < 0] = 0
      df[col] = df[col] / (self.csv_feature_dict[col]['MAX'] - self.csv_feature_dict[col]['MIN'])

    pad = np.zeros((self.max_len, len(df.columns)))
    length = min(len(df), self.max_len)
    pad[-length:] = df.to_numpy()[-length:]
    csv_feature = pad.T

    img = cv2.imread(img_path)
    if self.transform is not None:
      img = self.transform(image=img)['image']

    csv_feature = torch.tensor(csv_feature, dtype=torch.float32)
    label = torch.tensor(self.label_to_idx[label], dtype=torch.long)

    return {
      'img': img,
      'csv_feature': csv_feature,
      'label': label
    }

test_path = path + '/test'
class TestDataset(Dataset):
  def __init__(self, files, transform=None):
    self.files = files
    self.transform = transform

    self.csv_feature_dict = csv_feature_dict
    self.max_len = 196
    self.label_to_idx = label_to_idx

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):

    image_name = self.files[idx].split('/')[-1]
    img_path = f'{self.files[idx]}/{image_name}.jpg'
    csv_path = f'{self.files[idx]}/{image_name}.csv'

    df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
    df = df.replace('-', np.nan)
    df.fillna(method='post')
    try:
      for col in self.csv_feature_dict.keys():
        df[col] = df[col].astype('float32') - self.csv_feature_dict[col]['MIN']
        df[col][df[col] < 0] = 0
        df[col] = df[col] / (self.csv_feature_dict[col]['MAX'] - self.csv_feature_dict[col]['MIN'])
    except TypeError:
      print(image_name)

    pad = np.zeros((self.max_len, len(df.columns)))
    length = min(len(df), self.max_len)
    pad[-length:] = df.to_numpy()[-length:]
    csv_feature = pad.T

    img = cv2.imread(img_path)
    if self.transform is not None:
      img = self.transform(image=img)['image']

    csv_feature = torch.tensor(csv_feature, dtype=torch.float32)
    return {
      'img': img,
      'csv_feature': csv_feature,
    }


class CNN(nn.Module):
  def __init__(self, class_n, rate=0.1, use_pretrain=True):
    super(CNN, self).__init__()
    model = EfficientNet.from_name('efficientnet-b2')
    pretrained_dict = OrderedDict()
    model_dict = model.state_dict()

    checkpoint = torch.load(ck_dir, map_location=torch.device('cpu'))
    for k, v in checkpoint.items():
      name = k.replace("module.", "")
      pretrained_dict[name] = v

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # for n,p in model.named_parameters():
    #  if '_fc' not in n:
    #    p.requires_grad = False

    # model = torch.nn.parallel.DistributedDataParallel(model)

    self.model = model

  def forward(self, x):
    return self.model(x)

class LSTM(nn.Module):
  def __init__(self, max_len, emb_dim, num_features, class_n, rate):
    super(LSTM,self).__init__()

    self.lstm = nn.LSTM(max_len, emb_dim)
    self.rnn_fc = nn.Linear(num_features *  emb_dim , 1000)
    self.final_layer = nn.Linear(2000, class_n)
    self.dropout = nn.Dropout(rate)

  def forward(self, cnn_out, csv_input):
    hidden, _ = self.lstm(csv_input)
    hidden = hidden.view(hidden.size(0), -1) # (num_features * emb_dim , 1)
    hidden = self.rnn_fc(hidden)

    concat = torch.cat([cnn_out, hidden], dim = 1)
    return self.dropout(self.final_layer(concat))

class CNN2LSTM(nn.Module):
  def __init__(self, max_len ,emb_dim, num_features, class_n, rate):
    super(CNN2LSTM, self).__init__()

    self.cnn = CNN(class_n, rate)
    self.lstm = LSTM(max_len ,emb_dim, num_features, class_n, rate)

  def forward(self, img, seq):
    cnn_out = self.cnn(img)
    return self.lstm(cnn_out, seq)

def Train(epochs, model):

  best_score = -1
  model = model.to(device)

  for epoch in tqdm_notebook(range(epochs)):
    model.train()
    train_loss = []
    for data in train_loader:
      optimizer.zero_grad()
      img = data['img'].to(device)
      csv_feature = data['csv_feature'].to(device)
      label = data['label'].to(device)

      cost = model(img, csv_feature)
      loss = loss_fn(cost, label)
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
        csv_feature = data['csv_feature'].to(device)
        label = data['label'].to(device)

        cost = model(img, csv_feature)
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


def get_answer_csv(model):
  model.eval()

  pred = []
  with torch.no_grad():
    for data in test_loader:
      img = data['img'].to(device)
      csv_feature = data['csv_feature'].to(device)

      cost = model(img, csv_feature)
      pred.append(torch.argmax(cost, dim=1).cpu().numpy())

  pred = np.concatenate(pred, axis=1)
  ans = [idx_to_label(i) for i in pred]
  submission = pd.read_csv('./submssion.csv')
  submission.iloc[:, 1] = ans
  submission.to_csv(path + 'new_submission.csv', index=False)

  return submission



if __name__ == "__main__":
  fix(seed)

  ## Split train, valid
  train_df , val_df = split(train_label)

  ## Dataset, DataLoader
  train_dataset = LGDataset(file_df=train_df, transform=train_transform)
  val_dataset = LGDataset(file_df=val_df, transform=valid_transform)

  train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=2)
  val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=2)

  ## Esnemble Model
  model = CNN2LSTM(max_len, emb_dim, num_features, class_n, rate)

  ## Training parameter

  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam([
    {'params': model.cnn.parameters(), 'lr': cnn_lr},
    {'params': model.lstm.parameters()}
  ], lr=lr)
  lr_sc = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, verbose=True)

  ## Training
  Train(epochs, model)

  ## Test Dataset,Dataloader
  files = glob(test_path + '/*')
  test_dataset = TestDataset(files=files, transform=TTA)
  test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=2)

  ## Test & get submission csv files
  model.load_state_dict(torch.load(best_model_path))
  ans_csv = get_answer_csv(model)
