import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

class Concate():
    # TODO: change the encode_path naming to neg_path
    def __init__(self, pos_fasta_path, encode_path, label_path, encode_n, cell_cluster,
                 subset, test_small_cluster, exclude_index_path, include_index_path):
        self.pos_fasta_path = pos_fasta_path
        self.encode_path = encode_path
        self.label_path = label_path
        self.encode_n = encode_n
        self.cell_cluster = cell_cluster
        self.subset = subset
        self.test_small_cluster = test_small_cluster
        self.exclude_index = np.loadtxt(exclude_index_path, dtype = 'int')
        self.include_index = np.loadtxt(include_index_path, dtype = 'int')

    def concate_data(self):
        pos_fasta = pd.read_csv(self.pos_fasta_path,sep=">chr*",header=None, engine='python').values[1::2][:,0] # (n,) array, each element is string, dtype=object
        encode_fasta = pd.read_csv(self.encode_path,sep=">chr*",header=None, engine='python').values[1::2][:,0]
        label = pd.read_csv(self.label_path, delimiter = ",", header=None) ## TODO check these are np array

        print("-----------finish pd read_csv------------")
        print("-----------original shape------------")
        print("pos_size")
        print(pos_fasta.shape) #(114538,)
        print("neg_size")
        print(encode_fasta.shape) #(1127227,)
        print("label")
        print(label.shape) #(114538, 1)

        np.random.seed(202101190)
        encode_index = np.random.choice(encode_fasta.shape[0], size=self.encode_n, replace=False)
        encode_fasta = encode_fasta[encode_index]


        data = np.concatenate([pos_fasta, encode_fasta])
        #encode_n = int(encode_fasta.shape[0]) ## check if need to devide by 2

        # label = label.iloc[:,0].str.strip("{}").str.get_dummies(',').rename(
        #     columns=lambda x: x.strip("'"))
        # n_frag = int(label.shape[0])

        #label = label.to_numpy().reshape(n_frag, 8) ## shape is n x 8
        label = label.to_numpy()

        label = label[:, self.cell_cluster] ## select the cluster for label

        neg_label = np.zeros((self.encode_n, len(self.cell_cluster)))
        label = np.concatenate([label, neg_label])


        print("-----------shape right after encode concate------------")
        print(data.shape) #(124538,)
        print(label.shape) #(124538, 1)

        if self.test_small_cluster == "True":

            exclude_fasta = data[self.exclude_index]
            exclude_label = label[self.exclude_index]
            print("----------- test_small_cluster is true ------------")
            print("----------- exclude shape ------------")
            print("exclude_fasta")
            print(exclude_fasta.shape)
            print("exclude_label")
            print(exclude_label.shape)

            data = data[self.include_index]
            label = label[self.include_index]
            print("----------- test_small_cluster is true ------------")
            print("----------- shape ------------")
            print("pos_size")
            print(data.shape)
            print("label")
            print(label.shape)  # (114538, 1)

        else:
            exclude_fasta = np.empty([1])
            exclude_label = np.empty([1])
            print("-----------------fake exclude_fasta and exclude_label -------------")
            print(exclude_fasta)
            print(exclude_label)

        if self.subset == "True":
            data, label = self._subset(data, label)
        print("-----------------shape after subset -------------")
        print(data.shape)
        print(label.shape)


        print("-----------original pos/neg ratio------------")
        print(np.count_nonzero(label)/(label.shape[0]*label.shape[1]-np.count_nonzero(label)))
        pos_weight = []
        for i in range(0,label.shape[1]):
            num_pos = np.count_nonzero(label[:, i])
            num_neg = label.shape[0] - num_pos
            pos_weight.append(float(num_neg)/num_pos)
        print(pos_weight)


        return data, label, pos_weight, exclude_fasta, exclude_label

    def split_train_test(self, data, label, exclude_fasta, exclude_label, test_size = 0.1):
        data_train_temp, data_test, label_train_temp, label_test = train_test_split(data, label, test_size=test_size, random_state=12)
        data_train, data_eval, label_train, label_eval = train_test_split(data_train_temp, label_train_temp, test_size=test_size, random_state=12)

        print("-----------train size concate------------")
        print(data_train.shape) #(173775,)
        print(label_train.shape) #(173775, 1)

        print("-----------eval size concate------------")
        print(data_eval.shape) #(19309,)
        print(label_eval.shape) #(19309, 1)

        print("-----------test size concate------------")
        print(data_test.shape) #(21454,)
        print(label_test.shape) #(21454,)

        if self.test_small_cluster == "True":
            data_test = np.concatenate([data_test, exclude_fasta])
            label_test = np.concatenate([label_test, exclude_label])
            print("-----------test_small_cluster is true------------")
            print("-----------below is updated test label------------")
            print(data_test.shape)  # (21454,)
            print(label_test.shape)  # (21454,)


        return data_train, data_eval, data_test, label_train, label_eval, label_test

    def _subset(self, data, label):
        size = int(np.floor(data.shape[0] * 0.1))
        np.random.seed(202101190)
        index = np.random.choice(data.shape[0], size=size, replace=False)

        data = data[index]
        label = label[index, :]

        return data, label

    # TODO 04/12 add the balance option



class ConcateTrans():
    def __init__(self, pos_forward_path, neg_forward_path, balance, subset):
        self.pos_forward_path = pos_forward_path
        self.neg_forward_path = neg_forward_path
        self.balance = balance
        self.subset = subset

    def concate_data(self):
        pos_fasta = pd.read_csv(self.pos_forward_path, sep=">chr*",
                                header=None, engine='python').values[1::2][:, 0]  # (n,) array, each element is string, dtype=object
        neg_fasta = pd.read_csv(self.neg_forward_path, sep=">chr*",
                                header=None, engine='python').values[1::2][:, 0]

        print("-----------finish pd read_csv------------")
        print("-----------original shape------------")
        print("positive")
        print(pos_fasta.shape)
        print("negative")
        print(neg_fasta.shape)

        if self.balance == "True":
            pos_fasta, neg_fasta = self._balance(pos_fasta, neg_fasta)

        print("-----------positve/negative ratio after balance------------")
        print(pos_fasta.shape[0]/neg_fasta.shape[0])

        data = np.concatenate([pos_fasta, neg_fasta])

        n_pos = pos_fasta.shape[0]
        n_neg = neg_fasta.shape[0]

        pos_label = np.ones(n_pos)
        neg_label = np.zeros(n_neg)

        label = np.concatenate([pos_label, neg_label]).reshape((n_pos+n_neg), 1) # nx1

        print("-----------shape right after join pos and neg fasta------------")
        print(data.shape)  # (1681932,)
        print(label.shape)  # (1681932, 1)

        if self.subset == "True":
            data, label = self._subset(data, label)

        print("-----------------shape after subset -------------")
        print(data.shape)
        print(label.shape)

        print("-----------pos/neg ratio------------")
        print(np.count_nonzero(label) / (label.shape[0] * label.shape[1] - np.count_nonzero(label)))
        pos_weight = []
        for i in range(label.shape[1]):
            num_pos = np.count_nonzero(label[:, i])
            num_neg = label.shape[0] - num_pos
            pos_weight.append(float(num_neg) / num_pos)
        print(pos_weight)

        return data, label, pos_weight

    def split_train_test(self, data, label, test_size = 0.1):
        data_train_temp, data_test, label_train_temp, label_test = train_test_split(data, label, test_size=test_size, random_state=12)
        data_train, data_eval, label_train, label_eval = train_test_split(data_train_temp, label_train_temp, test_size=test_size, random_state=12)

        return data_train, data_eval, data_test, label_train, label_eval, label_test

    def _subset(self, data, label):
        size = int(np.floor(data.shape[0] * 0.1))
        np.random.seed(202101190)
        index = np.random.choice(data.shape[0], size=size, replace=False)

        data = data[index]
        label = label[index, :]

        return data, label

    def _balance(self, pos, neg):
        pos_size = pos.shape[0]
        neg_size = neg.shape[0]

        if pos_size > neg_size:
            ratio = neg_size/pos_size
            size = int(np.floor(pos.shape[0] * ratio))
            np.random.seed(202101190)
            index = np.random.choice(pos.shape[0], size=size, replace=False)
            pos = pos[index]
        else:
            ratio = pos_size / neg_size
            size = int(np.floor(neg.shape[0] * ratio))
            np.random.seed(202101190)
            index = np.random.choice(neg.shape[0], size=size, replace=False)
            neg = neg[index]

        return pos, neg


# TODO finish this
class Processor():
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def split_train_test(self, test_size = 0.1):
        data_train_temp, data_test, label_train_temp, label_test = train_test_split(self.data, self.label, test_size=test_size, random_state=12)
        data_train, data_eval, label_train, label_eval = train_test_split(data_train_temp, label_train_temp, test_size=test_size, random_state=12)

        return data_train, data_eval, data_test, label_train, label_eval, label_test


    # TODO give the data and label
    def _balance(self):
        pos_size = self.pos.shape[0]
        neg_size = self.neg.shape[0]

        if pos_size > neg_size:
            ratio = neg_size/pos_size
            size = int(np.floor(self.pos.shape[0] * ratio))
            np.random.seed(202101190)
            index = np.random.choice(self.pos.shape[0], size=size, replace=False)
            pos = self.pos[index]
        else:
            ratio = pos_size / neg_size
            size = int(np.floor(self.neg.shape[0] * ratio))
            np.random.seed(202101190)
            index = np.random.choice(self.neg.shape[0], size=size, replace=False)
            neg = self.neg[index]

        return pos, neg

# TODO make the concate two class, Concate and Concate_trans
# TODO make balance, subset, tran_test_split another class, as Processor