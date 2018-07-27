import os
from utils import PredictionsUtils as p_utils
from classifiers.RolesClassifier import RolesClassifier

ROLES_DATASET_PATH        = 'datasets/obs_roles_dataset.csv'
ROLES_FEATURE_COLUMNS     = ['CollectionVariables','AddListenerMethod','RemoveListenerMethod','ClassDeclarationKeyword',
                             'MethodsDeclarationKeyword','ClassType','ScanCollectionsMethod','SCMCallsAbsMethod','HasSuperclass',
                             'ImplementsInterfaces','ChangeState','AfterChangeStateIterateOverList','Role']
ROLES_LABELS              = ['Subject', 'Observer', 'ConcreteSubject', 'ConcreteObserver','None']

INSTANCES_DATASET_PATH    = "datasets/obs_instances_dataset.csv"
INSTANCES_FEATURE_COLUMNS = ['HasSubject','HasObserver','SubjectsRelationship','SubObsDependencies',
                             'CSubObsDependencies','ObserversRelationship','CallListeners','CObsAccessSubject',
                             'NoC','IsObserver']
INSTANCE_LABELS           = ['Not Observer','Observer']

PREDICTIONS_ROOT_DIRECTORY       = 'predictions'
ROLES_PREDICTIONS_HEADER         = ['Class','Role','Probability']
ROLES_PREDICTIONS_FILE_PATH      = PREDICTIONS_ROOT_DIRECTORY+'/roles_predictions.csv'
INSTANCES_COMBINATIONS_HEADER    = ['Classes']+INSTANCES_FEATURE_COLUMNS[:len(INSTANCES_FEATURE_COLUMNS)-1]
INSTANCES_COMBINATIONS_FILE_PATH = PREDICTIONS_ROOT_DIRECTORY+'/combinations.csv'

ROLES_TRAIN_BATCH_SIZE    = 20
ROLES_EVALUATE_BATCH_SIZE = 5
ROLES_TRAINING_STEPS      = 5000

INSTANCES_TRAIN_BATCH_SIZE    = 5
INSTANCES_EVALUATE_BATCH_SIZE = 2
INSTANCES_TRAINING_STEPS      = 1

FOLDERS_NUMBER = 5

def main():
    SW_PATH='softwares/TestSW.csv'

    SW_ROLES_BATCH_SIZE=8
    SW_CLASSES_COMBINATIONS_BATCH_SIZE=5

    rolesClassifier=RolesClassifier(ROLES_FEATURE_COLUMNS,ROLES_LABELS,FOLDERS_NUMBER)
    rolesClassifier.initFeatureColumns()
    rolesClassifier.initClassifier()
    rolesClassifier.loadDataset(ROLES_DATASET_PATH,0,';')
    rolesClassifier.suffleDataset()
    rolesClassifier.kFoldersTrainAndEvaluation(ROLES_TRAIN_BATCH_SIZE,ROLES_TRAINING_STEPS,ROLES_EVALUATE_BATCH_SIZE,True)

    '''
    instancesClassifier = InstancesClassifier(INSTANCES_FEATURE_COLUMNS, INSTANCE_LABELS, FOLDERS_NUMBER)
    instancesClassifier.initFeatureColumns()
    instancesClassifier.initClassifier()
    instancesClassifier.loadDataset(INSTANCES_DATASET_PATH, 0, ';')
    instancesClassifier.suffleDataset()
    instancesClassifier.kFoldersTrainAndEvaluation(INSTANCES_TRAIN_BATCH_SIZE,INSTANCES_TRAINING_STEPS,INSTANCES_EVALUATE_BATCH_SIZE,True)
    '''

    sw_classes,roles_predictions = rolesClassifier.predict(SW_PATH, 0, ';', SW_ROLES_BATCH_SIZE)
    predictions_list=p_utils.get_predictions_list(sw_classes,roles_predictions,ROLES_LABELS)
    p_utils.log_predictions_on_file(PREDICTIONS_ROOT_DIRECTORY,ROLES_PREDICTIONS_FILE_PATH,ROLES_PREDICTIONS_HEADER,predictions_list)

    classes_quadruplets, classes_triplets, classes_pairs=p_utils.roles_permutation(predictions_list)
    abs_abs_pairs, con_abs_pairs = p_utils.filter_pairs_list(predictions_list, classes_pairs)
    cs_obs_co_triplets = p_utils.filter_triplets_list(predictions_list, classes_triplets)

    combinations=[abs_abs_pairs, con_abs_pairs, cs_obs_co_triplets, classes_quadruplets]
    p_utils.log_combinations_on_file(INSTANCES_COMBINATIONS_FILE_PATH,INSTANCES_COMBINATIONS_HEADER,combinations)

    '''
    _,instances_predictions=instancesClassifier.predict(SW_CLASSES_COMBINATIONS_PATH,0,';',SW_CLASSES_COMBINATIONS_BATCH_SIZE)
    '''

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()