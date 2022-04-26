import torch

from utils import *
from GlobalConvLayer import *
from lab1 import *
import numpy as np
import matplotlib.pyplot as plt
import time


# generate every combination of spikes for a given length
def spikePerms(n):
    out = np.empty((2 ** n, n), dtype=np.float32)
    for i in range(2 ** n):
        s = np.binary_repr(i, n)
        out[i, :] = np.fromiter(map(lambda si: float('inf') if si == '0' else 0.0, s), dtype=np.float32)

    return out


# convert a tensor of (rfc x rf0 x rf1 x chans) to (rfc x rf0 x rf1 x (2chans))
def posnegify(tensor):
    negTensor = torch.zeros_like(tensor)
    negTensor[tensor != float('inf')] = float('inf')
    # interleve the positive tensor and the negitive tensor
    oShape = (*tensor.shape[:-1], 2 * tensor.shape[-1])
    return torch.stack((tensor, negTensor), dim=-1).view(oShape)


def main():
    nside = 2
    nbits = nside * nside
    ninpts = 2 ** nbits
    nneurons = 8
    nepochs = 500
    doposneg = True

    ucap = 0.5
    ubackoff = 1.25
    usearch = 1/512

    rng = np.random.default_rng(seed=54321)

    # setup some stuff
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    if not cuda:
        torch.set_num_threads(8)

    # make the data
    d = spikePerms(4)
    d = d.reshape((-1, 2, 2, 1))
    d = torch.as_tensor(d)
    if doposneg:
        d = posnegify(d)
    d = d.to(device)

    # make the layer
    # input size (1 side), RF size (1 side), RF stride, prev layer neurons, cur layer neurons, threshold
    layer = TNNColumnLayer(nside, nside, 1, 2 if doposneg else 1, nneurons, 4, ntype='rnl', device=device)

    if cuda:
        layer.cuda()

    # train the layer
    weightChanges = np.empty(nepochs)
    for epoch in range(nepochs):
        ts = time.time()
        ow = layer.weights.clone()
        for i in rng.permutation(d.shape[0]):
            td = d[i]
            o, li, lo = layer(td)
            layer.weights = layer.stdp(li, lo, layer.weights, ucap, usearch, ubackoff)
        dw = torch.linalg.vector_norm(layer.weights - ow)
        weightChanges[epoch] = dw

        print('epoch {} done in {:.2f}, weight change {:.2f}'.format(epoch, time.time() - ts, dw))

    # find which neuron each pattern activates
    neurInd = np.empty(ninpts, dtype=np.uint)
    for i, inp in enumerate(d):
        o, li, lo = layer(inp)
        res = torch.argmin(o)
        neurInd[i] = res

    # make a new neuron order that looks more like the identity matrix
    newOrder = np.ones(nneurons, dtype=np.int) * nneurons
    nextSpot = 0
    # for each input pattern, find the neuron it activates
    for i, ind in enumerate(neurInd):
        # if that neuron hasn't been assigned a spot yet, assign it the next available spot
        if newOrder[ind] >= nneurons:
            newOrder[ind] = nextSpot
            nextSpot += 1

    # find the neurons that haven't been assigned yet and just assign them in order (they were never activated)
    for i, x in enumerate(newOrder):
        if x >= nneurons:
            newOrder[i] = nextSpot
            nextSpot += 1

    mapBack = np.empty_like(newOrder)
    for i, ni in enumerate(newOrder):
        mapBack[ni] = i

    # arrange the weights in this new order
    sweights = layer.weights[mapBack, :]

    # generate the confusion matrix based on the new order
    cmat = np.zeros((nneurons, ninpts), dtype=np.uint8)
    for i, neuron in enumerate(neurInd):
        cmat[newOrder[neuron], i] = 1

    # save the confusion matrix
    plt.matshow(cmat)
    plt.ylabel('activated neuron')
    plt.xlabel('input sequence')
    plt.savefig('demux_confusion.png')

    # save the weights
    displayWeights(sweights, (1, 4), 'demux', posneg=doposneg, nrow=1)

    # plot the norm of the weight changes over each epoch
    plt.figure()
    plt.semilogy(weightChanges)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('norm of weight change over epoch')
    plt.tight_layout()
    plt.savefig('weightChanges.png')


if __name__ == '__main__':
    main()
