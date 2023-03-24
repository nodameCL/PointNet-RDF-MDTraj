import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
import json
import re
import pandas as pd 

class MDtrajRDFxyzvTypeNPZ(data.Dataset):
    def __init__(self,root, split='train'):
        self.root = root
        self.data_file = os.path.join(root,f'{split}_wPT.npz')
        self.data = np.load(self.data_file, allow_pickle=True)

        self.x = torch.from_numpy(self.data['x'][:, :, :3].astype(np.float32)) # xyz or xyzvType
        self.y = torch.from_numpy(self.data['y'].astype(np.float32))
        self.sys = torch.from_numpy(self.data['PT'].astype(np.float32))
        self.frame = torch.from_numpy(self.data['frame'].astype(np.float32).reshape(-1, 1))

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sys[index], self.frame[index]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    
    datapath = '../dataset/'

    d = MDtrajRDFxyzvTypeNPZ(root=datapath)
    print("length of dataset is:", len(d))

 



