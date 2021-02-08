import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# rea
# d and organize x and y
def read_data(x_path, x_r_path, y_path):
#    x_path="/Users/yanwengong/Documents/JingZhang_lab/snp_effect/data/pbmc_snapATAC/pre_deep_learning_TEST"
    X = pd.read_csv(x_path)
    x_r = pd.read_csv(x_r_path)
    y = pd.read_csv(y_path, sep='\t', header=None)

    return X, x_r, y


def add_reverse_complement(X, x_r, y):
    return pd.concat([X, x_r]), pd.concat([y, y])


def prepare_data_1D(X, y):
    n_frag = int(X.shape[0]/1000)
    X = X.values.reshape(-1, 1000, 4)

    y = y.iloc[:,4]
    y = y.str.strip("{}").str.get_dummies(',').rename(columns=lambda x: x.strip("'"))
    n_frag2 = y.shape[0]
    if n_frag != n_frag2:
        sys.exit("X and y dimension not match")
    y = y.to_numpy().reshape(n_frag2, 8)

    return X, y


def train_split(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=12)
    return X_train, X_test, y_train, y_test

#def modify_size(X_train, X_test, y_train, y_test, indices_train, indices_test):
#    X_train = X_train[0:35839, :]
#    X_test = X_test[8192, :]
#    y_train = y_train[0:35839, :]
#    y_test = y_test[8192, :]
#    indices_train


# def save_data(path, data, indices):
#     print(data.shape)
#     print(len(indices))
#     counter = 0
#     for i in range(data.shape[0]):
#         counter += 1
#         name = str(indices[i])
#         np.save(os.path.join(path, name), data[i])
#     print(counter)


def save_data(output_path_X, X_train, X_test, output_path_y, y_train, y_test):
    np.save(os.path.join(output_path_X, "X_train"), X_train)
    np.save(os.path.join(output_path_X, "X_test"), X_test)
    np.save(os.path.join(output_path_y, "y_train"), y_train)
    np.save(os.path.join(output_path_y, "y_test"), y_test)



def main(x_path, x_r_path, y_path, output_path_X, output_path_y):
    X, x_r, y = read_data(x_path, x_r_path, y_path)
    X, y = add_reverse_complement(X, x_r, y)
    X, y = prepare_data_1D(X, y)
    X_train, X_test, y_train, y_test = train_split(X, y, test_size=0.2)
    save_data(output_path_X, X_train, X_test, output_path_y, y_train, y_test)

#def save_x_train_test(x_path, x_r_path, y_path, x_train_test_path):
#    X, x_r, y = read_data(x_path, x_r_path, y_path)
#    X, y = add_reverse_complement(X, x_r, y)
#    X, y = prepare_data_1D(X, y)
#    X_train, X_test, y_train, y_test, indices_train, indices_test = train_split(X, y, test_size=0.2)
#    np.save(os.path.join(x_train_test_path, "X_train_data"), X_train)
#    np.save(os.path.join(x_train_test_path, "X_test_data"), X_test)

main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])