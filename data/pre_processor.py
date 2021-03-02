import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Processor():
    def __init__(self, pos_fasta_path, encode_path, label_path, encode_n):
        self.pos_fasta_path = pos_fasta_path
        self.encode_path = encode_path
        self.label_path = label_path
        self.encode_n = encode_n

    def concate_data(self):
        pos_fasta = pd.read_csv(self.pos_fasta_path,sep=">chr*",header=None, engine='python').values[1::2][:,0] # (n,) array, each element is string, dtype=object
        encode_fasta = pd.read_csv(self.pos_fasta_path,sep=">chr*",header=None, engine='python').values[1::2][:,0]
        label = pd.read_csv(self.label_path, delimiter = "\t", header=None) ## TODO check these are np array
        print("-----------finish pd read_csv------------")

        np.random.seed(202101190)
        encode_index = np.random.choice(encode_fasta.shape[0], size=self.encode_n, replace=False)
        encode_fasta = encode_fasta[encode_index]

        data = np.concatenate([pos_fasta, encode_fasta])
        #encode_n = int(encode_fasta.shape[0]) ## check if need to devide by 2

        label = label.iloc[:,0].str.strip("{}").str.get_dummies(',').rename(
            columns=lambda x: x.strip("'"))
        n_frag = int(label.shape[0])

        label = label.to_numpy().reshape(n_frag, 8) ## shape is n x 8
        #print(label.shape) # (114538, 8)

        neg_label = np.zeros((self.encode_n, 8))
        label = np.concatenate([label, neg_label])

        print("-----------shape right after concate------------")
        print(data.shape) #(214538,)
        print(label.shape) #(214538, 8)

        return data, label

    def split_train_test(self, data, label, test_size = 0.1):
        data_train_temp, data_test, label_train_temp, label_test = train_test_split(data, label, test_size=test_size, random_state=12)
        data_train, data_eval, label_train, label_eval = train_test_split(data_train_temp, label_train_temp, test_size=test_size, random_state=12)

        return data_train, data_eval, data_test, label_train, label_eval, label_test



class ProcessorTrans():
    def __init__(self, pos_forward_path, neg_forward_path):
        self.pos_forward_path = pos_forward_path
        self.neg_forward_path = neg_forward_path

    def concate_data(self):
        pos_fasta = pd.read_csv(self.pos_forward_path, sep=">chr*",
                                header=None, engine='python').values[1::2][:, 0]  # (n,) array, each element is string, dtype=object
        neg_fasta = pd.read_csv(self.neg_forward_path, sep=">chr*",
                                header=None, engine='python').values[1::2][:, 0]

        print("-----------finish pd read_csv------------")
        data = np.concatenate([pos_fasta, neg_fasta])

        n_pos = pos_fasta.shape[0]
        n_neg = neg_fasta.shape[0]

        pos_label = np.ones(n_pos)
        neg_label = np.zeros(n_neg)

        label = np.concatenate([pos_label, neg_label]).reshape((n_pos+n_neg), 1) # nx1

        print("-----------shape right after concate------------")
        print(data.shape)  # (214538,)
        print(label.shape)  # (214538, 8)
        return data, label

    def split_train_test(self, data, label, test_size = 0.1):
        data_train_temp, data_test, label_train_temp, label_test = train_test_split(data, label, test_size=test_size, random_state=12)
        data_train, data_eval, label_train, label_eval = train_test_split(data_train_temp, label_train_temp, test_size=test_size, random_state=12)

        return data_train, data_eval, data_test, label_train, label_eval, label_test


