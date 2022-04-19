import os
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

class DVSEvent():
    # initialize using byte array of 5 elements
    def __init__(self, bytearr):
        # store the 40 bit number as (39)b4(32)|(31)b3(24)|(23)b2(16)|(15)b1(8)|(7)b0(0)
        # b0, b1, b2, b3, b4 = f.read(5) # 40 bits or 8 bytes per event
        b4, b3, b2, b1, b0 = bytearr
        
        # bit 39 - 32: Xaddress (in pixels)
        # bit 31 - 24: Yaddress (in pixels)
        # bit 23: polarity (0 for OFF, 1 for ON)
        # bit 22 - 0: Timestamp (in microseconds)

        # 8 bits for x and y coordinates
        self.x = b4
        self.y = b3
        # 1 bit for polarity
        self.polarity = b2 >> 7
        
        # upper 7 bits of timestamp
        time_upper7 = b2 & 0x7f
        self.timestamp = (time_upper7 << 16) | (b1 << 8) | b0

    def print(self):
        print("x:%2d y:%2d polarity:%d timestamp:%6d" % (self.x, self.y, self.polarity, self.timestamp))

class DVSVideo():
    # initialize using file name
    def __init__(self, video_path, label):
        self.name = video_path.split('/')[-1]
        self.label = label
        self.events = []
        with open(video_path, 'rb') as f:
            bytearr = f.read(5)
            while bytearr:
                self.events.append(DVSEvent(bytearr))
                bytearr = f.read(5)

    # aggregate all events 
    def aggregate_events (self):

        # create np arrays
        pos_arr = np.zeros((60, 60))
        neg_arr = np.zeros((60, 60))

        # iterate through events
        for event in self.events:
            
            # add to pos and neg array
            if event.polarity:
                pos_arr[event.y, event.x] += 1
            else:
                neg_arr[event.y, event.x] += 1

        # per NMNIST, only 28x28 area is relevant
        posneg_arr = np.stack((pos_arr, neg_arr))
        posneg_arr = posneg_arr[:, 4:32, 4:32]
        return posneg_arr

    # return spiketime array by inverting number of events, and setting zeros to inf
    def spiketime_arr(self, thres):
        agg_events = np.clip(self.aggregate_events() - thres, 0, np.inf)
        max_events = np.amax(agg_events)

        spiketimes = max_events - agg_events
        spiketimes = np.where(spiketimes == max_events, np.inf, spiketimes)
        return spiketimes

class N_MNIST(Dataset):
    def __init__(self, dir, train, thres=0):
        suffix = 'Train' if train else 'Test'
        self.dir = dir + suffix + '/'
        self.videos = pd.read_csv(dir + 'annotations_' + suffix + '.csv', names=['video', 'label'])
        self.videos = self.videos.sort_values('video')
        self.thres = thres

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        dvs_path = os.path.join(self.dir, self.videos.iloc[idx, 0])
        label = self.videos.iloc[idx, 1]

        dvs_video = DVSVideo(dvs_path, label)
        spiketime_arr = torch.from_numpy(dvs_video.spiketime_arr(self.thres).astype(np.float32))
        return spiketime_arr, label


# testing script
# mode = 'Train'
# label = '0'
# dvs = DVSVideo('N-MNIST/Train/4/06888.bin', label)

# arr = dvs.spiketime_arr(7)
# print(arr.shape)
# plt.imshow(arr[0])
# plt.show()
# plt.imshow(arr[1])
# plt.show()