from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import gc

datalabel = "TNF"

def data_label():
    return datalabel

def MCNN_data_load(DATA_TYPE, MAXSEQ):
    """
    Load positive and negative data for MCNN and shuffle them.
    DATA_TYPE: placeholder (not used)
    MAXSEQ: placeholder (not used)
    Returns: x_train, y_train, x_test, y_test
    """

    TNF_train = "C:/jupyter/juan/TNF/DATASET/pos/prollma/newest_prollma.npy"
    TNF_test = "C:/jupyter/juan/TNF/DATASET/pos/test/pos_test.npy"
    Cytokine_train = "C:/jupyter/juan/TNF/DATASET/neg/neg_train.npy"
    Cytokine_test = "C:/jupyter/juan/TNF/DATASET/neg/neg_test.npy"

    # Load and shuffle training data
    x_train, y_train = data_load(TNF_train, Cytokine_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # Load and shuffle testing data
    x_test, y_test = data_load(TNF_test, Cytokine_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    return x_train, y_train, x_test, y_test


def data_load(pos, neg):
    """
    Load positive and negative .npy files and create one-hot labels.
    """
    pos_file = np.load(pos)
    neg_file = np.load(neg)

    pos_label = np.ones(pos_file.shape[0])
    neg_label = np.zeros(neg_file.shape[0])

    x = np.concatenate([pos_file, neg_file], axis=0)
    y = np.concatenate([pos_label, neg_label], axis=0)
    y = tf.keras.utils.to_categorical(y, 2)

    gc.collect()
    return x, y
