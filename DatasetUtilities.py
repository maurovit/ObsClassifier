import tensorflow as tf
import itertools

KEYS_PREFIX="folder_"

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
    quadruplets_list = list(itertools.permutations(predictions_list, 4))
    triplets_list = list(itertools.permutations(predictions_list, 3))
    pairs_list = list(itertools.permutations(predictions_list, 2))

    return (quadruplets_list, triplets_list, pairs_list)