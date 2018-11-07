import os
from classifiers.InstancesClassifier import InstancesClassifier
from utils import PredictionsUtils as p_utils

INSTANCES_FEATURE_COLUMNS = ['HasSubject','HasObserver','SubjectsRelationship','SubObsDependencies',
                             'CSubObsDependencies','ObserversRelationship','CallListeners','CObsAccessSubject',
                             'NoC','IsObserver']
INSTANCES_LABELS          = ['Not Observer','Observer']
INSTANCES_DATASET_PATH    = "../datasets/obs_instances_dataset.csv"

FOLDERS_NUMBER = 5

INSTANCES_TRAIN_BATCH_SIZE    = 5
INSTANCES_EVALUATE_BATCH_SIZE = 2
INSTANCES_TRAINING_STEPS      = 500

PREDICTIONS_ROOT_DIRECTORY       = '../predictions'
INSTANCES_MOKUP_PATH             = PREDICTIONS_ROOT_DIRECTORY +'/combinations_mokup.csv'
INSTANCES_PREDICTIONS_FILE_PATH  = PREDICTIONS_ROOT_DIRECTORY +'/instances_predictions.csv'
INSTANCES_PREDICTIONS_HEADER     = ['Combinations','Result','Probability']


INSTANCES_COMBINATIONS_HEADER    = ['Classes'] + INSTANCES_FEATURE_COLUMNS[:len(INSTANCES_FEATURE_COLUMNS)-1]
INSTANCES_COMBINATIONS_FILE_PATH = PREDICTIONS_ROOT_DIRECTORY + '/combinations_to_test.csv'
INSTANCES_MOKUP_PATH             = PREDICTIONS_ROOT_DIRECTORY +'/combinations_mokup.csv'
INSTANCES_PREDICTIONS_FILE_PATH  = PREDICTIONS_ROOT_DIRECTORY +'/instances_predictions.csv'
INSTANCES_PREDICTIONS_HEADER     = ['Combinations','Result','Probability']

def main():
    SW_CLASSES_COMBINATIONS_BATCH_SIZE = 5

    instancesClassifier = InstancesClassifier(INSTANCES_FEATURE_COLUMNS, INSTANCES_LABELS, FOLDERS_NUMBER)
    instancesClassifier.initFeatureColumns()
    instancesClassifier.initClassifier()
    instancesClassifier.loadDataset(INSTANCES_DATASET_PATH, 0, ';')
    instancesClassifier.suffleDataset()
    instancesClassifier.kFoldersTrainAndEvaluation(INSTANCES_TRAIN_BATCH_SIZE, INSTANCES_TRAINING_STEPS,INSTANCES_EVALUATE_BATCH_SIZE, True)
    instances, instances_predictions = instancesClassifier.predict(INSTANCES_MOKUP_PATH, 0, ';',SW_CLASSES_COMBINATIONS_BATCH_SIZE)
    instances_predictions_list = p_utils.get_instances_predictions_list(instances, instances_predictions,INSTANCES_LABELS)
    p_utils.log_predictions_on_file(PREDICTIONS_ROOT_DIRECTORY, INSTANCES_PREDICTIONS_FILE_PATH,INSTANCES_PREDICTIONS_HEADER, instances_predictions_list)
    print("Output has been produced in predictions folder")

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()