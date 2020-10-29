#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:34:29 2020

@author: pasq
"""
import os
import numpy as np
import pandas as pd
import time

import torch
from torch import nn
import torchvision

path = r"/home/pasq/crop-yield-prediction-project"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def diff(x):
    x = x.to_numpy()
    x_0 = np.concatenate([x, np.array([0])])
    x_1 = np.concatenate([np.array([0]), x])
    diff = x_1[1:-1]/x_0[1:-1]-1
    return diff

def labelling(path, type_opt):
    yields = pd.read_csv(path)
    commodity = yields.loc[yields["Type"] == type_opt]
    time_series = commodity.groupby(["State", "County ANSI", "Year"]).sum().unstack(-1)
    time_series_2019 = time_series.iloc[:,-2:]
    commo_clean = time_series_2019.loc[~(time_series_2019.isna().sum(1)>0)]
    comm_diff = commo_clean.apply(diff, axis=1)
    comm_diff = pd.DataFrame(comm_diff.to_list(), index=comm_diff.index, columns=["Yield"])
    return comm_diff

#Restrict y to list_np
def lab_selecting(labels, npy_folder):
    list_npy = os.listdir(npy_folder)
    cs_in_fold = pd.Series(list_npy).str[:-4].str.split("_")
    cs_in_fold = pd.DataFrame(cs_in_fold.to_list())
    cs_in_fold[0] = "s"+cs_in_fold[0]
    cs_in_fold[1] = "c"+cs_in_fold[1]
    new_labels = labels.loc[labels.index.isin(cs_in_fold.to_numpy().tolist())]
    return new_labels


class EEDataset(torch.utils.data.Dataset):
    def __init__(self, path, labels, transform=False):
        self.path = path
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels.iloc[idx][0]
        s,c = self.labels.iloc[idx].name
        s,c = s[1:], c[1:]
        fname = s+"_"+c+".npy"
        img = np.load(path+"/numpy_data/"+fname)
        img = img[:,:6,:,:]
        if self.transform:
            mu, sigma = self.transform
            if np.random.rand() < .5:
                img = np.fliplr(img)
            if np.random.rand() < .5:
                img = np.flipud(img)
            
            img = (img.copy()-mu)/sigma
        
        sample = {"x": img, "y": label}
        return sample



corn_labs = labelling(path+r"/2019_clean_data/2019_yield_final.csv", "C")
corn_select = lab_selecting(corn_labs, path+r"/numpy_data")

mu = np.array((8.41852082e+02, 8.23091039e+02, 6.71061327e+02, 3.37676476e-01))
sigma = np.array((1.16219494e+03, 1.09720981e+03, 1.09359154e+03, 3.21207187e-01))
mu = mu[:, np.newaxis, np.newaxis, np.newaxis]
sigma = sigma[:, np.newaxis, np.newaxis, np.newaxis]

#Comparison between normalized and original images
dataset = EEDataset(path, corn_select, transform=(mu,sigma))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#dataiter = iter(dataloader)
#
#batch = next(dataiter)
#x = batch["x"]
#img = x[0,:3,-1,:,:].numpy()
#img = np.transpose(img, (1,2,0))
#plt.imshow(img/10000)
#
#dataset = EEDataset(path, corn_select, transform=(mu, sigma))
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#dataiter_norm = iter(dataloader)
#
#batch = next(dataiter_norm)
#x = batch["x"]
#img_norm = x[0,:3,-1,:,:].numpy()
#img_norm = np.transpose(img_norm, (1,2,0))
#plt.imshow(img_norm)

from tcn import TemporalConvNet

tcn = TemporalConvNet.to(device)
loss_fn = nn.SmoothL1Loss(reduction="mean", beta=.1)
optimizer = torch.optim.Adam(tcn.parameters(), lr=1e-4)


def train(model, loss_fn, optimizer, loader_train, epochs=1):    
    loss_ts = []
    start = time.time()
    for epoch in range(epochs):
        for i, batch in enumerate(loader_train):
            model.train()
            xb, yb = batch.values()
            xb, yb = xb.float().to(device), yb.float().to(device)
            y_hat = model(xb)
            loss = loss_fn(y_hat, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_ts.append(loss.item())
            
            model.eval()
            print(f"Epoch/Iteration: {epoch}/{i} | Loss: {loss.item()}")
            if i%5 == 0:
                torch.save(model.state_dict(), path+"model_state.pth")
            
            print("Time elapsed: %d" % (time.time()-start))
            start = time.time()
    return loss_ts

ts = train(tcn, loss_fn, optimizer, dataloader)
        
