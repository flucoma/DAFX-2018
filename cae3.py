#!/usr/bin/env python3
import sys, os
import numpy as np
import torch
from os.path import splitext
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
        self.deconv2 = nn.ConvTranspose2d(4, 2, 5, 1,2)
        kaiming_uniform(self.conv1.weight)
        kaiming_uniform(self.deconv2.weight)

    def forward(self, x):
        intitial_size = x.size()
        x = self.conv1(x)
        x = F.relu(x)
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

def process_file(input_fname, l1, l2, l3):
    print(l1,l2,l3)
    w = Wave.read(input_fname)
    X = stft.process(w)
    ae = AutoEncoder()
    ae.load_state_dict(torch.load("ae_4x2_poly.pickle"))

    if torch.cuda.is_available(): ae.cuda()
    criterion  = torch.nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), weight_decay = 0.5)
    v = get_features(X)
    for epoch in range (n_iterations):
        epoch_loss = 0
        ae.train()
        optimizer.zero_grad()
        output = ae(v)
        out1 = output[0,0,:,:]
        out2 = output[0,1,:,:]
        mix = out1 + out2
        tgt = v[0,0,:,:]
        mse = criterion(mix,tgt)
        hdiff1 = torch.sum(torch.pow(out1[:,1:] - out1[:,:-1], 2))/(torch.pow(torch.norm(out1,2),2)+eps)
        vdiff1 = torch.sum(torch.pow(out1[1:,:] - out1[:-1,:], 2))/(torch.pow(torch.norm(out1,2),2)+eps)
        hdiff2 = torch.sum(torch.pow(out2[:,1:] - out2[:,:-1], 2))/(torch.pow(torch.norm(out2,2),2)+eps)
        vdiff2 = torch.sum(torch.pow(out2[1:,:] - out2[:-1,:], 2))/(torch.pow(torch.norm(out2,2),2)+eps)
        loss1 = torch.norm(mix,1)
        tloss = hdiff1/(vdiff1+eps)
        sloss = vdiff2/(hdiff2+eps)
        print(
            "mse {:>10.10f} loss 1 {:>10.10f} T loss {:>10.10f} S loss {:>10.10f}".format(
                    mse.data.item(), loss1.data.item(), tloss.data.item(), sloss.data.item()
            )
        )
        loss = mse + l1 * loss1 + l2 * tloss + l3 * sloss
        loss.backward()
        optimizer.step()
    O = ae(v)
    O1 = O[0,0,:,:].cpu().data.numpy()
    O2 = O[0,1,:,:].cpu().data.numpy()

    T = O1.T
    S = O2.T

    Smask = S/(S+T+eps)
    Tmask = T/(S+T+eps)

    steady =  istft.process(Spectrogram(X*Smask, X.sample_rate))
    trans =  istft.process(Spectrogram(X*Tmask, X.sample_rate))
    steady.write(splitext(input_fname)[0]+"_steady.wav")
    trans.write(splitext(input_fname)[0]+"_trans.wav")

if len(sys.argv) != 5:
    print("usage: cae3.py in_file l1 l2 l3")
else:
    process_file(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
