import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 128, 3, 2)
        self.bn4 = nn.BatchNorm1d(128, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(640, 128)
        self.linear2 = nn.Linear(128, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = x.view(x.shape[0], -1)

        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x
  
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 128, 3, 2)
        self.bn4 = nn.BatchNorm1d(128, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(1408, 128)
        self.linear2 = nn.Linear(128, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 128, 3, 2)
        self.bn4 = nn.BatchNorm1d(128, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(2176, 128)
        self.linear2 = nn.Linear(128, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 128, 3, 2)
        self.bn4 = nn.BatchNorm1d(128, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(3072, 128)
        self.linear2 = nn.Linear(128, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 128, 3, 2)
        self.bn4 = nn.BatchNorm1d(128, momentum=0.997)

        #self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(3840, 128)
        self.linear2 = nn.Linear(128, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # exit()
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x
