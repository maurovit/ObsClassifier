import os
import DatasetUtilities as utils
from RolesClassifier import RolesClassifier
from InstancesClassifier import InstancesClassifier

ROLES_DATASET_PATH        = "datasets/obs_roles_dataset.csv"
ROLES_FEATURE_COLUMNS     = ['CollectionVariables','AddListenerMethod','RemoveListenerMethod','ClassDeclarationKeyword',
                             'MethodsDeclarationKeyword','ClassType','ScanCollectionsMethod','SCMCallsAbsMethod','HasSuperclass',
                             'ImplementsInterfaces','ChangeState','AfterChangeStateIterateOverList','Role']
ROLES_LABELS              = ['Subject', 'Observer', 'ConcreteSubject', 'ConcreteObserver','None']

INSTANCES_DATASET_PATH    = "datasets/obs_instances_dataset.csv"
INSTANCES_FEATURE_COLUMNS = ['HasSubject','HasObserver','SubjectsRelationship','SubObsDependencies',
                             'CSubObsDependencies','ObserversRelationship','CallListeners','CObsAccessSubject',
                             'NoC','IsObserver']
INSTANCE_LABELS           = ['Not Observer','Observer']


ROLES_TRAIN_BATCH_SIZE    = 20
ROLES_EVALUATE_BATCH_SIZE = 5
ROLES_TRAINING_STEPS      = 5000

INSTANCES_TRAIN_BATCH_SIZE    = 5
INSTANCES_EVALUATE_BATCH_SIZE = 2
INSTANCES_TRAINING_STEPS      = 550

FOLDERS_NUMBER = 5

def main():
    SW_PATH='softwares/software_sample.csv'
    SW_BATCH_SIZE=8

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

    sw_classes,predictions = rolesClassifier.predict(SW_PATH, 0, ';', SW_BATCH_SIZE)
    predictions_list=utils.get_prediction_list(sw_classes,predictions,ROLES_LABELS)
    classes_quadruplets,classes_triplets,classes_pairs=utils.roles_permutation(predictions_list)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()