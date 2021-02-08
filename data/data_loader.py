import numpy as np
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, x_data_path, y_data_path, cell_cluster, subset ="True"):
        self.X = np.moveaxis(np.load(x_data_path),1,2)
        self.y = np.load(y_data_path)[:, cell_cluster]
        self._subset = subset

        if self._subset == "True":
            print("subset dataset")

            size = int(np.floor(self.X.shape[0] * 0.005))
            np.random.seed(202101190)
            X_index = np.random.choice(self.X.shape[0], size=size, replace=False)
            np.random.seed(202101190)
            y_index = np.random.choice(self.y.shape[0], size=size, replace=False)

            self.X = self.X[X_index, :]
            self.y = self.y[y_index, :]

        print("----------shape--------------")
        print(self.X.shape)
        print(self.y.shape)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]

        return X, y


    # def get_x(self):
    #     X = np.load(self._x_data_path)
    #     return X
    #
    # def get_y(self):
    #     y = np.load(self._y_data_path)

        # if self._subset == "True":
        #     train_size = int(np.floor(len(indices_train) * 0.2))
        #     test_size = int(np.floor(len(indices_test) * 0.2))
        #     np.random.seed(202101190)
        #     indices_train = np.random.choice(indices_train, size=(train_size,), replace=False)
        #     np.random.seed(202101190)
        #     indices_test = np.random.choice(indices_test, size=(test_size,), replace=False)
        #     print("------------indices shape-------------")
        #     print("subset")
        #     print(indices_train.shape)
        #     print(indices_test.shape)
        #     print("------------indices shape-------------")
        #     pass

        # if self._exact_divide == "True":
        #     # train_size = int(182272)
        #     # test_size = int(45056)
        #     # np.random.seed(20210131)
        #     # indices_train = np.random.choice(indices_train, size=(train_size,), replace=False)
        #     # np.random.seed(20210131)
        #     # indices_test = np.random.choice(indices_test, size=(test_size,), replace=False)
        #     # print("------------indices shape-------------")
        #     # print("exact_divide")
        #     # print(indices_train.shape)
        #     # print(indices_test.shape)
        #     # print("------------indices shape-------------")
        #     pass
        #
        # return y
