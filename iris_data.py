import pandas as pd
import tensorflow as tf

TRAIN_PATH = "observer_dp_dataset.csv"
TEST_PATH= "observer_dp_dataset_test.csv"
VAL_PATH="observer_dp_dataset_validation.csv"

CSV_COLUMN_NAMES = ['CollectionVariables','AddListenerMethod','RemoveListenerMethod',
                    'ClassDeclarationKeyword','MethodsDeclarationKeyword','ClassType','ScanCollectionsMethod',
                    'SCMCallsAbsMethod','HasSuperclass','ImplementsInterfaces','ChangeState',
                    'AfterChangeStateIterateOverList','Role']
ROLES = ['Subject', 'Observer', 'ConcreteSubject', 'ConcreteObserver']

def load_data(y_name='Role'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0, delimiter=';')
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES, header=0, delimiter=';')
    test_x, test_y = test, test.pop(y_name)
    validation = pd.read_csv(VAL_PATH, names=CSV_COLUMN_NAMES, header=0, delimiter=';')
    val_x, val_y = validation, validation.pop(y_name)

    return (train_x, train_y), (test_x,test_y), (val_x,val_y)

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
