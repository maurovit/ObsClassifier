import pandas as pd
import tensorflow as tf

TRAIN_PATH = "obs_instances_dataset.csv"
COLUMN_NAMES=['HasSubject','HasObserver','SubjectsRelationship','SubObsDependencies','CSubObsDependencies',
              'ObserversRelationship','CallListeners','CObsAccessSubject','NoC','IsObserver']
ROLES=['Not Observer','Observer']

def load_data(y_name='IsObserver'):
    dataset=pd.read_csv(TRAIN_PATH,names=COLUMN_NAMES,header=0,delimiter=';')
    train_x, train_y= dataset,dataset.pop(y_name)
    k_folder(dataset)
    return (train_x,train_y)

def k_folder(dataset):
    count=1
    size=(int)(dataset['HasSubject'].count()/5)
    print("SIZE",size)
    folder=[];
    for index, row in dataset.iterrows():
        if(count<=5):
            folder.append((index,row))
        count+=1
    print(folder);

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

def main():
    x,y=load_data()

main()