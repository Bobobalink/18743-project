### Importing libraries ###

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GlobalConvLayer import *


import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image

from lab1 import *
from utils import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time


### Argument parsing ###

parser = argparse.ArgumentParser(description="A Temporal Neural Network Simulator")

parser.add_argument('--mode',        type=int, default=1,\
                    help='3 modes available: 0 - SNL column simulation; 1 - RNL Column Simulation; 2 - ECVT network simulation')

args = parser.parse_args()


### Weight Update Parameters ###

ucapture = 1.0 / 4
usearch = 1.0 / 3096
ubackoff = 1.0 / 4
rfsize = 5
neurons = 24
### Column Layer Parameters ###

inputsize = 32
stride    = 1
nprev     = 7
theta     = 4

### Voter Layer Parameters ###

rows_v = 20#inputsize + 1 - rfsize
cols_v = 20#inputsize + 1 - rfsize
nprev_v = neurons
classes_v = 10
thetav_lo = 1 / 32
thetav_hi = 15 / 32
tau_eff = 2

### Enabling CUDA support for GPU ###
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not cuda:
    torch.set_num_threads(8)

### MNIST dataset loading and preprocessing ###

train_loader = DataLoader(CIFAR10('./data', True, download=True, transform=transforms.Compose(
                                                                    [
                                                                     #transforms.Grayscale(num_output_channels=1),
                                                                     #transforms.ToTensor(),
                                                                     PosNegRGB(0.5, 
                                                                               transforms.Grayscale(num_output_channels=1),
                                                                               transforms.ToTensor())
                                                                    ]
                                                                                       )
                               ),
                          batch_size=1,
                          shuffle=False
                         )

test_loader = DataLoader(CIFAR10('./data', False, download=True, transform=transforms.Compose(
                                                                    [
                                                                     #transforms.Grayscale(num_output_channels=1),
                                                                     #transforms.ToTensor(),
                                                                     PosNegRGB(0.5, 
                                                                               transforms.Grayscale(num_output_channels=1),
                                                                               transforms.ToTensor())
                                                                    ]
                                                                                       )
                               ),
                          batch_size=1,
                          shuffle=False
                         )


############################## Online Learning ##############################

inc_learn   = 1
breakpoint1 = 60000
interval1   = 1000
breakpoint2 = 10000
interval2   = 1000

### Layer Initialization ###
# input size (1 side), RF size (1 side), RF stride, prev layer neurons, cur layer neurons, threshold
clayer = GlobalConvLayer(inputsize, rfsize, stride, nprev, 24, 4, ntype="rnl", device=device)
# clayer1 = TNNColumnLayer(28, 3, 1, 24, 24, 4, ntype="rnl", device=device)
# clayer2 = TNNColumnLayer(26, 3, 1, 24, 48, 4, ntype="rnl", device=device)
# clayer3 = TNNColumnLayer(24, 5, 1, 48, 48, 4, ntype="rnl", device=device)
# clayer4 = TNNColumnLayer(20, 5, 1, 48, 48, 4, ntype="rnl", device=device)
vlayer = DualTNNVoterTallyLayer(28, 28, 24, classes_v, thetav_lo, thetav_hi, tau_eff, device=device)

if cuda:
    clayer.cuda()
    # clayer1.cuda()
    # clayer2.cuda()
    clayer3.cuda()
    clayer4.cuda()
    vlayer.cuda()


### Training ###

for epochs in range(1):
    start = time.time()
    error1 = 0
    error2 = 0
    errorlist1 = []
    errorlist2 = []

    for idx, (data, target) in enumerate(train_loader):
        # if idx == breakpoint1:
        #     break
        print("Sample: {0}\r".format(idx+1), end="")

        if cuda:
            data                    = data.cuda()
            target                  = target.cuda()

        # if (idx+1) > 20000:
        #     data                    = torch.transpose(data,2,3)

        # if (idx+1) > 50000:
        #     if (target[0]%2) == 0:
        #         target[0] = target[0] + 1
        #     else:
        #         target[0] = target[0] - 1

        out, layer_in, layer_out    = clayer(data[0].permute(1, 2, 0))
        # out1, layer_in1, layer_out1 = clayer1(out)
        # out2, layer_in2, layer_out2 = clayer2(out1)
        # out3, layer_in3, layer_out3 = clayer3(out2)
        # out4, layer_in4, layer_out4 = clayer4(out3)
        pred, voter_in, _           = vlayer(out)

        if torch.argmax(pred) != target[0]:
            error1 += 1
            error2 += 1

        clayer.weights = clayer.stdp(layer_in, layer_out, clayer.weights, ucapture, usearch, ubackoff)
        # clayer1.weights = clayer1.stdp(layer_in1, layer_out1, clayer1.weights, ucapture, usearch, ubackoff)
        # clayer2.weights = clayer2.stdp(layer_in2, layer_out2, clayer2.weights, ucapture, usearch, ubackoff)
        # clayer3.weights = clayer3.stdp(layer_in3, layer_out3, clayer3.weights, ucapture, usearch, ubackoff)
        # clayer4.weights = clayer4.stdp(layer_in4, layer_out4, clayer4.weights, ucapture, usearch, ubackoff)
        vlayer.weights = vlayer.stdp(target, voter_in, vlayer.weights)

        if (idx+1)%interval1 == 0:
            errorlist1.append(error1/(idx+1))
            errorlist2.append(error2/interval1)
            error2 = 0

        endt = time.time()
        #print("                                                     Time elapsed: {0}\r".format(endt-start), end="")

    end = time.time()
    print("Training for {0} samples done in {1}".format(idx, end-start))
    print("Training Accuracy for {1} epochs: {0}%".format((60000-error1)*100/60000, epochs+1))


### Testing ###

    error3 = 0
    start = time.time()

    for idx, (data, target) in enumerate(test_loader):
        # if idx == breakpoint2:
        #     break
        print("Sample: {0}\r".format(idx+1+breakpoint1), end="")

        if cuda:
            data                    = data.cuda()
            target                  = target.cuda()

        # data                    = torch.transpose(data,2,3)
        # if (target[0]%2) == 0:
        #     target[0] = target[0] + 1
        # else:
        #     target[0] = target[0] - 1

        out, layer_in, layer_out    = clayer(data[0].permute(1, 2, 0))
        # out1, layer_in1, layer_out1 = clayer1(out)
        # out2, layer_in2, layer_out2 = clayer2(out1)
        # out3, layer_in3, layer_out3 = clayer3(out2)
        # out4, layer_in4, layer_out4 = clayer4(out3)
        pred, voter_in, _           = vlayer(out)


        if torch.argmax(pred) != target[0]:
            error1 += 1
            error2 += 1
            error3 += 1

        if inc_learn == 1:
            clayer.weights = clayer.stdp(layer_in, layer_out, clayer.weights, ucapture, usearch, ubackoff)
            # clayer1.weights = clayer1.stdp(layer_in1, layer_out1, clayer1.weights, ucapture, usearch, ubackoff)
            # clayer2.weights = clayer2.stdp(layer_in2, layer_out2, clayer2.weights, ucapture, usearch, ubackoff)
            # clayer3.weights = clayer3.stdp(layer_in3, layer_out3, clayer3.weights, ucapture, usearch, ubackoff)
            # clayer4.weights = clayer4.stdp(layer_in4, layer_out4, clayer4.weights, ucapture, usearch, ubackoff)
            vlayer.weights = vlayer.stdp(target, voter_in, vlayer.weights)

        if (idx+1)%interval2 == 0:
            errorlist1.append(error1/(idx+1+breakpoint1))
            errorlist2.append(error2/interval2)
            error2 = 0

        endt = time.time()
        #print("                                                     Time elapsed: {0}\r".format(endt-start), end="")

    end = time.time()
    print("Testing for {0} samples done in {1}".format(idx, end-start))
    print("Testing Accuracy: {0}%".format((10000-error3)*100/10000))

# save the weight images
image_list = []
for i in range(clayer.weights.shape[0]):
    rflen = clayer.rfsize[0] * clayer.rfsize[1]
    nchans = clayer.weights.shape[1] // rflen
    temp = clayer.weights[i].reshape(nchans * clayer.rfsize[0], clayer.rfsize[1])
    image_list.append(temp)
out = torch.stack(image_list, dim=0).unsqueeze(1)
save_image(out, 'weightviz.png', nrow=6, pad_value=0.25)


### Storing image ###
plt.figure(figsize=(10,4))

dx = [i for i in range(1,int(breakpoint1/interval1+breakpoint2/interval2+1))]
plt.plot(dx, errorlist2, color='red', linestyle='solid', linewidth=1.5)

plt.xlabel("Samples (x1000)")
plt.ylabel("Error Rate")
plt.xticks(np.arange(1,int(breakpoint1/interval1+breakpoint2/interval2+1),2))
plt.legend(["ECVT"])
plt.savefig("online_learning.png")
