import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from ast import literal_eval

class AirlineDataset(Dataset):
    def __init__(self, file_path, label_map_path):
        super().__init__()
        df = pd.read_csv(file_path)
        with open(label_map_path, 'rb') as f:
            self.label_map = pkl.load(f)
        
        df['label'] = df['intent'].map(self.label_map)
        self.labels = df['label'].to_list()
        self.label_onehot_mtx = [F.one_hot(torch.tensor(l), num_classes=len(self.label_map)) for l in self.labels]
        self.label_onehot_mtx = np.stack(self.label_onehot_mtx)
        self.data = []
 
        for i, row in df.iterrows():
            self.data.append((row['pid'], row['utterance'], row['response'], row['intent']))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pid, uttr, response, str_label = self.data[idx]
        label = self.label_onehot_mtx[idx]
        return (pid, uttr, response, label, str_label)

    def get_labels_num(self):
        return len(self.label_map)

class WozDataset(Dataset):
    def __init__(self, file_path, label_map_path):
        super().__init__()
        df = pd.read_csv(file_path)
        with open(label_map_path, 'rb') as f:
            self.label_map = pkl.load(f)
        self.label_multihot = []
        self.data = []
 
        for i, row in df.iterrows():
            l = torch.zeros(len(self.label_map))
            for intent in literal_eval(row['intent']):
                l[self.label_map[intent]] = 1
            self.label_multihot.append(l)
            self.data.append((row['pid'], row['utterance'], row['response'], row['intent']))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pid, uttr, response, str_label = self.data[idx]
        label = self.label_multihot[idx]
        return (pid, uttr, response, label, str_label)

    def get_labels_num(self):
        return len(self.label_map)

if __name__ == '__main__':
    dataset = WozDataset('data/woz_train.csv', 'data/woz_label_map.pkl', mode=None, factory=None)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(train_dataloader):
        pid, uttr, subcat, response, label, str_label = data[0], list(data[1]), list(data[2]), list(data[3]), data[4], list(data[7])
        print(data)
        break
