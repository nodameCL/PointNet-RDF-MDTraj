from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# from model.point_dataset import SCMDataset

class PointNetOne(nn.Module):
    def __init__(self, l1 = 64, l2 = 128, l3=1024, l4 = 512, l5 = 512, 
                 dropout = 0, isBN = False, isLN = False, pooling = 'max', liquid = 'Ar'):
        super(PointNetOne, self).__init__()

        self.liquid = liquid 
        if self.liquid == 'Ar':
            self.conv1 = torch.nn.Conv1d(6, l1, 1)
        else: 
            self.conv1 = torch.nn.Conv1d(8, l1, 1)

        self.ln1 = nn.LayerNorm([l1,56])
        self.conv2 = torch.nn.Conv1d(l1, l2, 1)
        self.conv3 = torch.nn.Conv1d(l2, l3, 1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.ln2 = nn.LayerNorm([l2,56])
        self.ln3 = nn.LayerNorm([l3,56])
        self.l3 = l3
        
        self.fc1 = nn.Linear(l3, l4)
        self.fc2 = nn.Linear(l4, l5)
        if self.liquid == 'Ar': 
            self.fc3 = nn.Linear(l5, 323)
        elif self.liquid == 'NO': 
            self.fc3 = nn.Linear(l5, 1140)
        else: 
            self.fc3 = nn.Linear(l5, 1240)
            
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(l4) 
        self.bn2 = nn.BatchNorm1d(l5)
        self.ln1 = nn.LayerNorm(l4)
        self.ln2 = nn.LayerNorm(l5)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        self.isBN = isBN
        self.isLN = isLN
        self.pooling = pooling 

    def forward(self, x):
        # step 1 learn features from point cloud 
        if self.isBN and not self.isLN: 
            x = self.relu1(self.bn1(self.conv1(x))) # x: [BZ, l1, num_points]
            x = self.relu2(self.bn2(self.conv2(x))) # x: [BZ, l2, num_points]
            x = self.bn3(self.conv3(x))         # x: [BZ, l3, num_points]
        elif self.isLN and not self.isBN: 
            x = self.relu1(self.ln1(self.conv1(x)))
            x = self.relu2(self.ln2(self.conv2(x)))
            x = self.ln3(self.conv3(x))
        else: 
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.conv3(x)

        # step 2 maxpooling the learned features 
        if self.pooling == 'max': 
            x = torch.max(x, 2, keepdim=True)[0] # x: [BZ, l3, 1]
        else: 
            x = torch.mean(x, 2, keepdim=True)
            
        # step 2.2 reshape the maxpooled features for step 3
        x = x.view(-1, self.l3) # x: [BZ, l3]

        # step 3 feed extracted features to FC layers 
        if self.isBN and not self.isLN: 
            x = self.relu3(self.bn1(self.fc1(x)))
            x = self.relu4(self.bn2(self.dropout(self.fc2(x))))
        elif self.isLN and not self.isBN: 
            x = self.relu3(self.ln1(self.fc1(x)))
            x = self.relu4(self.ln2(self.dropout(self.fc2(x))))
        else: 
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.dropout(self.fc2(x)))

        x = self.fc3(x)
        return x 

class PointNetOneXY(nn.Module):
    def __init__(self, l1 = 64, l2 = 128, l3=1024, l4 = 512, l5 = 512, 
                 dropout = 0, isBN = False, isLN = False, addTP = True, pooling = 'max', liquid = 'Ar'):
        super(PointNetOneXY, self).__init__()

        self.liquid = liquid 
        self.addTP = addTP 
        
        if self.liquid == 'Ar':
            self.conv1 = torch.nn.Conv1d(3, l1, 1)#(6, l1, 1)
        else: 
            self.conv1 = torch.nn.Conv1d(3, l1, 1) #(8, l1, 1)

        self.ln1 = nn.LayerNorm([l1,56])
        self.conv2 = torch.nn.Conv1d(l1, l2, 1)
        self.conv3 = torch.nn.Conv1d(l2, l3, 1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.bn3 = nn.BatchNorm1d(l3)
        self.ln2 = nn.LayerNorm([l2,56])
        self.ln3 = nn.LayerNorm([l3,56])
        self.l3 = l3
        
        if self.addTP: 
            self.fc1 = nn.Linear(l3 + 2, l4)
        else: 
            self.fc1 = nn.Linear(l3, l4)
            
        self.fc2 = nn.Linear(l4, l5)
        
        if self.liquid == 'Ar': 
            self.fc3 = nn.Linear(l5, 323)
        elif self.liquid == 'NO': 
            self.fc3 = nn.Linear(l5, 1140)
        else: 
            self.fc3 = nn.Linear(l5, 1240)
            
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(l4) 
        self.bn2 = nn.BatchNorm1d(l5)
        self.ln1 = nn.LayerNorm(l4)
        self.ln2 = nn.LayerNorm(l5)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        self.isBN = isBN
        self.isLN = isLN
        self.pooling = pooling 

    def forward(self, x, y):
        # step 1 learn features from point cloud 
        if self.isBN and not self.isLN: 
            x = self.relu1(self.bn1(self.conv1(x))) # x: [BZ, l1, num_points]
            x = self.relu2(self.bn2(self.conv2(x))) # x: [BZ, l2, num_points]
            x = self.bn3(self.conv3(x))         # x: [BZ, l3, num_points]
        elif self.isLN and not self.isBN: 
            x = self.relu1(self.ln1(self.conv1(x)))
            x = self.relu2(self.ln2(self.conv2(x)))
            x = self.ln3(self.conv3(x))
        else: 
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.conv3(x)

        # step 2 maxpooling the learned features 
        if self.pooling == 'max': 
            x = torch.max(x, 2, keepdim=True)[0] # x: [BZ, l3, 1]
        else: 
            x = torch.mean(x, 2, keepdim=True)
            
        # step 2.2 reshape the maxpooled features for step 3
        x = x.view(-1, self.l3) # x: [BZ, l3]
        
        if self.addTP: 
            x = torch.cat([x, y], 1)

        # step 3 feed extracted features to FC layers 
        if self.isBN and not self.isLN: 
            x = self.relu3(self.bn1(self.fc1(x)))
            x = self.relu4(self.bn2(self.dropout(self.fc2(x))))
        elif self.isLN and not self.isBN: 
            x = self.relu3(self.ln1(self.fc1(x)))
            x = self.relu4(self.ln2(self.dropout(self.fc2(x))))
        else: 
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.dropout(self.fc2(x)))

        x = self.fc3(x)
        return x 

if __name__ == '__main__':

    sim_data = Variable(torch.rand(128,3,2500)) # BZ = 40, 
    sim_scalar = Variable(torch.rand(128,3))

    cls = PointNetOneXY()
    out = cls(sim_data, sim_scalar)
    # print('class', out.size(), out)
    # print(cls)