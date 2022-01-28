'''
Dacon LG AI Contest
https://dacon.io/competitions/official/235870/overview/description

image, sequence input, Multi label classification

'''

import os
import json
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

## Label
train_label = pd.read_csv('./train.csv')
label_kind = np.unique(train_label.label.values).tolist()
label_to_idx = {i: idx for idx, i in enumerate(label_kind)}
idx_to_label = {idx: i for idx, i in enumerate(label_kind)}

## Dict for MinMaxScaling of sequence data
with open(path + '/feature_min_max.json', 'r') as f:
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
path = 'C:/Users/park1/PycharmProjects/Dacon'
class LGDataset(Dataset):
  def __init__(self, file_df):
    self.file_df = file_df
    self.path = path

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
    df = df.replace('-', 0)

    for col in self.csv_feature_dict.keys():
      df[col] = df[col].astype(float) - self.csv_feature_dict[col]['MIN']
      df[col][df[col] < 0] = 0
      df[col] = df[col] / (self.csv_feature_dict[col]['MAX'] - self.csv_feature_dict[col]['MIN'])

    pad = np.zeros((self.max_len, len(df.columns)))
    length = min(len(df), self.max_len)
    pad[-length:] = df.to_numpy()[-length:]
    csv_feature = pad.T

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255
    img = np.transpose(img, (2, 0, 1))

    img = torch.tensor(img, dtype=torch.float32)
    csv_feature = torch.tensor(csv_feature, dtype=torch.float32)
    label = torch.tensor(self.label_to_idx[label], dtype=torch.long)

    return {
      'img': img,
      'csv_feature': csv_feature,
      'label': label
    }

test_path = path + '/test'
class TestDataset(Dataset):
  def __init__(self, path):
    self.path = path

    self.csv_feature_dict = csv_feature_dict
    self.max_len = 196
    self.label_to_idx = label_to_idx

  def __len__(self):
    return len(os.listdir(self.path))

  def __getitem__(self, idx):
    image_name = sorted(os.listdir(self.path))[idx]
    img_path = os.path.join(self.path, f'{image_name}/{image_name}.jpg')
    csv_path = os.path.join(self.path, f'{image_name}/{image_name}.csv')
    df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
    df = df.replace('-', 0)
    for col in self.csv_feature_dict.keys():
      df[col] = df[col] - self.csv_feature_dict[col]['MIN']
      df[col][df[col] < 0] = 0
      df[col] = df[col] / (self.csv_feature_dict[col]['MAX'] - self.csv_feature_dict[col]['MIN'])

    pad = np.zeros((self.max_len, len(df.columns)))
    length = min(len(df), self.max_len)
    pad[-length:] = df.to_numpy()[-length:]
    csv_feature = pad.T

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255
    img = np.transpose(img, (2, 0, 1))

    img = torch.tensor(img, dtype=torch.float32)
    csv_feature = torch.tensor(csv_feature, dtype=torch.float32)
    return {
      'img': img,
      'csv_feature': csv_feature,
    }

class CNN(nn.Module):
  def __init__(self, class_n, rate = 0.1):
    super(CNN,self).__init__()
    self.model = models.resnet50(pretrained=True)

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
        val_true.append(label.cpu().numpy())
        val_pred.append(torch.argmax(cost, dim=1).cpu().numpy())

    lr_sc.step(np.mean(val_loss))
    val_true = np.concatenate(val_true, axis=0).astype(np.float32)
    val_pred = np.concatenate(val_pred, axis=0).astype(np.float32)
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
  train_dataset = LGDataset(train_df)
  val_dataset = LGDataset(val_df)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

  ## Esnemble Model
  model = CNN2LSTM(max_len, emb_dim, num_features, class_n, rate)

  ## Training parameter
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  lr_sc = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, verbose=True)


  ## Training
  Train(epochs, model)

  ## Test Dataset,Dataloader
  test_dataset = TestDataset(test_path)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  ## Test & get submission csv files
  model.load_state_dict(torch.load(best_model_path))
  ans_csv = get_answer_csv(model)

