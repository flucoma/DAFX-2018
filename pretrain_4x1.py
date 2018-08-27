import sys
import glob, random
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn.init import kaiming_uniform
from untwist.data import Wave
from untwist.transforms import STFT, ISTFT
import torch.nn.functional as F
from random import shuffle


n_iterations = 100
n_files = 100
n_weights = 1
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

def get_features(path, fr = None, to = None):
    w = Wave.read(path)
    w = w[fr:to,0]
    X = stft.process(w)
    f = np.abs(X).T
    M = f[np.newaxis,np.newaxis,:,:]
    v = Variable(torch.from_numpy(M.astype(np.float32)))
    if torch.cuda.is_available():
        v = v.cuda()
    return v

def pre_train_model():
   features = [get_features(f, from_sample, to_sample) for f in glob.glob(training_glob_pattern)]
   #files = glob.glob(training_glob_pattern)
   #for i in range(len(files)):
   #    f = files[i]
   #    features.append(get_features(f, from_sample, to_sample))
   ae = AutoEncoder()
   if torch.cuda.is_available(): ae.cuda()
   criterion  = torch.nn.MSELoss()
   optimizer =  optim.Adam(ae.parameters(), weight_decay = 0.01)
   for epoch in range (n_iterations):
       print(epoch,"/",n_iterations)
       epoch_loss = 0
       ae.train()
       for point in features:
           optimizer.zero_grad()
           output = ae(point)
           loss = criterion(output,point)
           loss.backward()
           optimizer.step()
           epoch_loss += loss.data.item()
       shuffle(features)
   ae.cpu()
   torch.save(ae.state_dict(), out_fname)



if len(sys.argv) < 3:
    print("usage: pretrain_4x1.py 'training_glob_pattern' out_fname [from_sample to_sample]")
    sys.exit()

training_glob_pattern = sys.argv[1]

# examples of training_glob_pattern:
# '../apple_loops/*.wav'
# '/path/to/dataset/*/*.wav'

out_fname = sys.argv[2]

from_sample = None
to_sample = None

if len(sys.argv) > 3:
    from_sample = int(sys.argv[3])
    to_sample = int(sys.argv[4])

pre_train_model()
