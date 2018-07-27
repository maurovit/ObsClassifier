import tensorflow as tf
import itertools
import logging
import sys
import csv
import os

KEYS_PREFIX='folder_'
QUADRUPLETS_PREFIX='quadruplets'
TRIPLETS_PREFIX='triplets'
PAIRS_PREFIX='pairs'

def k_folders(dataset,columnName,foldersNumber):
    dataset_size=dataset[columnName].count()
    folder_size=(int)(dataset_size/foldersNumber)
    all_folders=dict([])

    for i in range(foldersNumber):
        d=None
        initial_row = i * folder_size

        if((i+1)==foldersNumber):
            d=dataset.iloc[initial_row:, :]
        else:
            final_row=(i+1)*folder_size
            d=dataset.iloc[initial_row:final_row,:]

        all_folders.update({KEYS_PREFIX + str(i + 1): d})

    return all_folders;

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset