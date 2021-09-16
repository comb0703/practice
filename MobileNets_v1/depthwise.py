# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HPe8PZJtMtzahkL7hGZC873QfQOgPHyN
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("error")

#dataset
transformer = transforms.Compose([transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transformer
)
test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transformer

)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=128,
                                         shuffle=False,
                                         num_workers=4)

class CNN_depthwise(nn.Module):
 
    def __init__(self):
      
        super(CNN_depthwise, self).__init__()

        def conv_dw(inp, oup):
              return nn.Sequential(
                  # dw
                  nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                  nn.BatchNorm2d(inp),
                  nn.ReLU(inplace=True),

                  # pw
                  nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                  nn.BatchNorm2d(oup),
                  nn.ReLU(inplace=True),
                  )
              
        self.conv_layer = nn.Sequential(

            conv_dw(3,32),
            conv_dw(32,64),
            conv_dw(64,128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            conv_dw(128,256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16384, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):

        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x

net = CNN_depthwise()
net = net.to(device)

learning_rate = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

# train, test function
def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, prediction = outputs.max(1)

        total += labels.size(0)
        correct += prediction.eq(labels).sum().item()

        if batch_idx % 100 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current train accuracy:', str(prediction.eq(labels).sum().item() / labels.size(0)))
            print('Current train average loss:', loss.item() / 100)


    print('\nTrain accuarcy:', 100. * correct / total)
    print('Train average loss:', train_loss / total)


def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)

        outputs = net(images)
        loss += criterion(outputs, labels).item()

        _, prediction = outputs.max(1)
        correct += prediction.eq(labels).sum().item()

    print('\nTest accuarcy:', 100. * correct / total)
    print('Test average loss:', loss / total)


    state = {
        'net': net.state_dict()
    }

    file_name = 'CNN_depthwise.pt'
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved')

for epoch in range(0, 10):
    train(epoch)
    test(epoch)