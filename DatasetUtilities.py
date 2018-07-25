import tensorflow as tf
import itertools
import logging
import sys

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

def get_prediction_list(data, predictions, labels):
    i = 0
    classNames = []
    predictionResults = dict()

    for row, column in data.iterrows():
        classNames.append(row[1])

    for pred in predictions:
        class_id = pred['class_ids'][0]
        probability = pred['probabilities'][class_id]
        if (class_id != 4):
            predictionResults.update({classNames[i]: (labels[class_id], probability * 100)})
        i = i + 1

    return predictionResults

def roles_permutation(predictions_list):
    triplets_list = list(itertools.permutations(predictions_list, 3))
    pairs_list = list(itertools.permutations(predictions_list, 2))
    abs_permutation_list = []
    con_permutation_list = []
    for item in pairs_list:
        roleOne = predictions_list[item[0]][0]
        roleTwo = predictions_list[item[1]][0]
        if (roleOne == 'Subject' and roleTwo == 'Observer')or(roleOne == 'Observer' and roleTwo == 'Subject') :
            abs_permutation_list.append(item)
        elif roleOne == 'Subject' and roleTwo == 'Subject':
            abs_permutation_list.append(item)
        elif roleOne == 'Observer' and roleTwo == 'Observer':
            abs_permutation_list.append(item)
        elif(roleOne == 'ConcreteSubject' and roleTwo == 'ConcreteObserver')or(roleOne == 'ConcreteObserver' and roleTwo == 'ConcreteSubject') :
            con_permutation_list.append(item)
        elif roleOne == 'ConcreteSubject' and roleTwo == 'ConcreteSubject':
            con_permutation_list.append(item)
        elif roleOne == 'ConcreteObserver' and roleTwo == 'ConcreteObserver':
            con_permutation_list.append(item)
    quadruplets_list = list(itertools.product(abs_permutation_list, con_permutation_list))

    return (quadruplets_list, triplets_list, pairs_list)

def filter_pairs_list(prediction_list, pairs_list):
    abs_abs_pairs = []
    con_abs_pairs = []
    for item in pairs_list:
        roleOne = prediction_list[item[0]][0]
        roleTwo = prediction_list[item[1]][0]
        if (roleOne == 'Subject' and roleTwo == 'Observer')or(roleOne == 'Observer' and roleTwo == 'Subject') :
            abs_abs_pairs.append(item)
        elif roleOne == 'Subject' and roleTwo == 'Subject':
            abs_abs_pairs.append(item)
        elif roleOne == 'Observer' and roleTwo == 'Observer':
            abs_abs_pairs.append(item)
        elif roleOne == 'ConcreteSubject' and roleTwo == 'Observer':
            con_abs_pairs.append(item)
        elif roleOne == 'ConcreteSubject' and roleTwo == 'Subject':
            con_abs_pairs.append(item)
        elif roleOne == 'ConcreteObserver' and roleTwo == 'Subject':
            con_abs_pairs.append(item)
        elif roleOne == 'ConcreteObserver' and roleTwo == 'Observer':
            con_abs_pairs.append(item)

    return (abs_abs_pairs, con_abs_pairs)

def filter_triplets_list(prediction_list, triplets_list):
    index=0

    for item in triplets_list:
        roleOne = prediction_list[item[0]][0]
        roleTwo = prediction_list[item[1]][0]
        roleThree = prediction_list[item[2]][0]

        if roleOne == roleTwo and roleTwo == roleThree:
            del triplets_list[index]
        elif roleTwo == 'ConcreteSubject' or roleTwo == 'ConcreteObserver':
            del triplets_list[index]
        elif roleOne == 'Subject' or roleOne == 'Observer':
            del triplets_list[index]
        elif roleThree == 'Subject' or roleThree == 'Observer':
            del triplets_list[index]

        index+=1

    return triplets_list


def get_logger(format):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=format)
    logger=logging.getLogger()
    return logger

def log_predictions():
    logger=get_logger('%(message)s')
    return

def log_permutations():
    logger=get_logger('%(message)s')
    return

