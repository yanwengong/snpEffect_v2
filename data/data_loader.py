import numpy as np


class DataLoader:

    def __init__(self, x_data_path, y_data_path, subset ="True", exact_divide ="True"):
        self._x_data_path = x_data_path
        self._y_data_path = y_data_path
        self._subset = subset
        self._exact_divide = exact_divide

    def get_x(self):
        return np.load(self._x_data_path)

    def get_y(self):
        y = np.load(self._y_data_path)

        if self._subset == "True":
            # train_size = int(np.floor(len(indices_train) * 0.2))
            # test_size = int(np.floor(len(indices_test) * 0.2))
            # np.random.seed(202101190)
            # indices_train = np.random.choice(indices_train, size=(train_size,), replace=False)
            # np.random.seed(202101190)
            # indices_test = np.random.choice(indices_test, size=(test_size,), replace=False)
            # print("------------indices shape-------------")
            # print("subset")
            # print(indices_train.shape)
            # print(indices_test.shape)
            # print("------------indices shape-------------")
            pass

        if self._exact_divide == "True":
            # train_size = int(182272)
            # test_size = int(45056)
            # np.random.seed(20210131)
            # indices_train = np.random.choice(indices_train, size=(train_size,), replace=False)
            # np.random.seed(20210131)
            # indices_test = np.random.choice(indices_test, size=(test_size,), replace=False)
            # print("------------indices shape-------------")
            # print("exact_divide")
            # print(indices_train.shape)
            # print(indices_test.shape)
            # print("------------indices shape-------------")
            pass

        return y
