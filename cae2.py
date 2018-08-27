#!/usr/bin/env python3
import sys, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn.init import kaiming_uniform
from untwist.data import Wave, Spectrogram
from untwist.transforms import STFT, ISTFT
import torch.nn.functional as F

n_iterations = 100
eps = np.spacing(1)
stft = STFT('hann',1024,256)
istft =  ISTFT('hann',1024,256)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 1, 2)
        self.deconv2 = nn.ConvTranspose2d(4, 1, 5, 1,2)
        kaiming_uniform(self.conv1.weight)
        kaiming_uniform(self.deconv2.weight)

    def forward(self, x):
        intitial_size = x.size()
        x = self.conv1(x)
        x = F.relu(x)
        size_after_conv1 = x.size()
        x = self.deconv2(x, output_size = intitial_size)
        x = F.relu(x)
        return x

def get_features(X):
    f = np.abs(X).T
    M = f[np.newaxis,np.newaxis,:,:]
    v = Variable(torch.from_numpy(M.astype(np.float32)))
    if torch.cuda.is_available():
        v = v.cuda()
    return v

def get_diffs(x):
    hdiff = torch.sum(torch.pow(x[:,1:] - x[:,:-1], 2))/(torch.pow(torch.norm(x,2),2)+eps)
    vdiff = torch.sum(torch.pow(x[1:,:] - x[:-1,:], 2))/(torch.pow(torch.norm(x,2),2)+eps)
    return hdiff, vdiff

def process_file(input_fname, output_fname, l1, l2, direction):
    w = Wave.read(input_fname)
    print(w.dtype)

    X = stft.process(w)
    ae = AutoEncoder()
    ae.load_state_dict(torch.load("ae_4x1_poly.pickle"))
    if torch.cuda.is_available(): ae.cuda()
    criterion  = torch.nn.MSELoss()
    optimizer =  optim.Adam(ae.parameters(), weight_decay = 0.01)
    v = get_features(X)
    for epoch in range (n_iterations):
        epoch_loss = 0
        ae.train()
        optimizer.zero_grad()
        output = ae(v)
        hdiff, vdiff = get_diffs(output[0,0,:,:])
        if (direction == 0):
            tsloss = hdiff/(vdiff+eps)
        else:
            tsloss = vdiff/(hdiff+eps)
        mse = criterion(output,v)
        loss1 =  torch.norm(output,1)
        loss2 =  tsloss
        print("mse {:>10.10f} loss 1 {:>10.10f} loss 2 {:>10.10f}".format(
            mse.data.item(), loss1.data.item(), loss2.data.item())
        )
        loss = mse + l1 * loss1 + l2 * loss2
        loss.backward()
        optimizer.step()
    O = ae(v)
    O = O[0,0,:,:].cpu().data.numpy()
    Y = np.abs(O.T) * np.exp(np.angle(X)*1j)
    y = istft.process(Spectrogram(Y, X.sample_rate))
    y.write(output_fname)

if len(sys.argv) != 6:
    print("usage: cae2.py in_file out_file l1 l2 direction")
else:
    process_file(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
