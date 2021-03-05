import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime

class Data(Dataset):
    def __init__(self, data, label, cell_cluster, subset ="True"):
        # (n,) array, each element is string, dtype=object
        self.data = data # fasta of forward, no chr title, 1d np.array, shape is n
        #self.label = label[:, cell_cluster] TODO commented on 03/03
        self.label = label
        print("-----------------shape before add RC -------------")
        print(self.data.shape)
        print(self.label.shape)

        if subset == "True":
            self._subset()
        print("-----------------shape after subset -------------")
        print(self.data.shape)
        print(self.label.shape)

        # add reverse complement
        temp = []
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N' : 'N'}
        print("reverse complement start time: ", datetime.now())
        for seq in self.data:
            complement_seq = ''
            for base in seq: ## need to check here, what is seq's shape?? why do they have seq[0] here
                complement_seq = complement[base] + complement_seq

            temp.append(complement_seq)# 0301 indented, TODO: check why before it could still train, even x and y have different n

        print("reverse complement end time: ", datetime.now())
        temp = np.array(temp, dtype=object)
        self.data = np.append(self.data, temp, axis=0)
        self.label = np.append(self.label, self.label, axis=0)
        print("-----------------shape after init data loader-------------")
        print(self.data.shape)
        print(self.label.shape)
        # if self.label.shape[1] == 1:
        #     pos_count = np.count_nonzero(self.label)
        #     neg_count = self.data.shape[0] - np.count_nonzero(self.label)
        #     self.weight_value = neg_count/pos_count
        #     print(f"calculated weight is {self.weight_value}")


    def __len__(self):
        return self.data.shape[0] ## check

    def __getitem__(self, index):
        seq = self.data[index]
        row_index = 0
        temp = np.zeros((len(seq), 4))
        for base in seq: ## seq[0]??
            if base == 'A':
                temp[row_index, 0] = 1
            elif base == 'T':
                temp[row_index, 1] = 1
            elif base == 'G':
                temp[row_index, 2] = 1
            elif base == 'C':
                temp[row_index, 3] = 1
            row_index += 1

        X = torch.tensor(temp).float().permute(1,0) # change the dim to 4, 1000
        y = torch.tensor(self.label[index]).float()

        return X, y

    def _subset(self):
        size = int(np.floor(self.data.shape[0] * 0.1))
        np.random.seed(202101190)
        X_index = np.random.choice(self.data.shape[0], size=size, replace=False)
        np.random.seed(202101190)
        y_index = np.random.choice(self.label.shape[0], size=size, replace=False)

        self.data = self.data[X_index]
        self.label = self.label[y_index, :]
