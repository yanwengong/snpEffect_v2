import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# rea
# d and organize x and y
def read_data(x_path, x_r_path, x_encode_path, y_path):
#    x_path="/Users/yanwengong/Documents/JingZhang_lab/snp_effect/data/pbmc_snapATAC/pre_deep_learning_TEST"
    X = pd.read_csv(x_path)
    x_r = pd.read_csv(x_r_path)
    x_encode = pd.read_csv(x_encode_path)
    encode_n = int(x_encode.shape[0]/1000)
    y = pd.read_csv(y_path, sep='\t', header=None)

    return X, x_r, x_encode, y, encode_n


def add_reverse_complement(X, x_r, x_encode, y):
    return pd.concat([X, x_r, x_encode]), pd.concat([y, y])


def prepare_data_1D(X, y, encode_n):
    n_frag = int(X.shape[0]/1000)
    X = X.values.reshape(-1, 1000, 4)

    y = y.iloc[:,4]
    y = y.str.strip("{}").str.get_dummies(',').rename(columns=lambda x: x.strip("'")) ## this is one-hot encoding of y
    n_frag2 = int(y.shape[0])

    y = y.to_numpy().reshape(n_frag2, 8)
    y_encode = np.zeros((encode_n, 8))
    y = np.concatenate((y, y_encode), axis=0)

    n_frag2 = y.shape[0]
    print(n_frag)
    print(n_frag2)

    if n_frag != n_frag2:
       sys.exit("X and y dimension not match")


    ## TODO add the encode y label as all zero

    return X, y


def train_split(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=12)
    return X_train, X_test, y_train, y_test


def save_data(output_path_X, X_train, X_test, output_path_y, y_train, y_test):
    np.save(os.path.join(output_path_X, "X_train"), X_train)
    np.save(os.path.join(output_path_X, "X_test"), X_test)
    np.save(os.path.join(output_path_y, "y_train"), y_train)
    np.save(os.path.join(output_path_y, "y_test"), y_test)



def main(x_path, x_r_path, x_encode_path, y_path, output_path_X, output_path_y):
    X, x_r, x_encode, y, encode_n = read_data(x_path, x_r_path, x_encode_path, y_path)
    X, y = add_reverse_complement(X, x_r, x_encode, y)
    X, y = prepare_data_1D(X, y, encode_n)
    X_train, X_test, y_train, y_test = train_split(X, y, test_size=0.2)
    save_data(output_path_X, X_train, X_test, output_path_y, y_train, y_test)


main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])