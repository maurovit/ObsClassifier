import os
from utils import PredictionsUtils as p_utils
from classifiers.RolesClassifier import RolesClassifier
from tester import InstancesClassifierTester

ROLES_DATASET_PATH        = '../datasets/obs_roles_dataset.csv'
ROLES_FEATURE_COLUMNS     = ['CollectionVariables','AddListenerMethod','RemoveListenerMethod','ClassDeclarationKeyword',
                             'MethodsDeclarationKeyword','ClassType','ScanCollectionsMethod','SCMCallsAbsMethod','HasSuperclass',
                             'ImplementsInterfaces','ChangeState','AfterChangeStateIterateOverList','Role']
ROLES_LABELS              = ['Subject', 'Observer', 'ConcreteSubject', 'ConcreteObserver','None']

ROLES_TRAIN_BATCH_SIZE    = 20
ROLES_EVALUATE_BATCH_SIZE = 5
ROLES_TRAINING_STEPS      = 5500

FOLDERS_NUMBER = 5

PREDICTIONS_ROOT_DIRECTORY       = '../predictions'
SOFTWARES_ROOT_DIRECTORY         = '../softwares'
ROLES_PREDICTIONS_HEADER         = ['Class','Role','Probability']
ROLES_PREDICTIONS_FILE_PATH      = PREDICTIONS_ROOT_DIRECTORY + '/roles_predictions.csv'
SW_PATH                          = SOFTWARES_ROOT_DIRECTORY+'/TestSoftware.csv'

def main():
    SW_ROLES_BATCH_SIZE=8

    rolesClassifier=RolesClassifier(ROLES_FEATURE_COLUMNS,ROLES_LABELS,FOLDERS_NUMBER)
    rolesClassifier.initFeatureColumns()
    rolesClassifier.initClassifier()
    rolesClassifier.loadDataset(ROLES_DATASET_PATH,0,';')
    rolesClassifier.suffleDataset()
    rolesClassifier.kFoldersTrainAndEvaluation(ROLES_TRAIN_BATCH_SIZE,ROLES_TRAINING_STEPS,ROLES_EVALUATE_BATCH_SIZE,True)

    sw_classes,roles_predictions = rolesClassifier.predict(SW_PATH, 0, ';', SW_ROLES_BATCH_SIZE)
    roles_predictions_list=p_utils.get_roles_predictions_list(sw_classes,roles_predictions,ROLES_LABELS)
    p_utils.log_predictions_on_file(PREDICTIONS_ROOT_DIRECTORY,ROLES_PREDICTIONS_FILE_PATH,ROLES_PREDICTIONS_HEADER,roles_predictions_list)

    classes_quadruplets, classes_triplets, classes_pairs = p_utils.roles_permutation(roles_predictions_list)
    abs_abs_pairs, con_abs_pairs = p_utils.filter_pairs_list(roles_predictions_list, classes_pairs)
    cs_obs_co_triplets = p_utils.filter_triplets_list(roles_predictions_list, classes_triplets)

    combinations = [abs_abs_pairs, con_abs_pairs, cs_obs_co_triplets, classes_quadruplets]
    p_utils.log_combinations_on_file(InstancesClassifierTester.INSTANCES_COMBINATIONS_FILE_PATH, InstancesClassifierTester.INSTANCES_COMBINATIONS_HEADER, combinations)

    print('The combinations to test as observer instances are in '+InstancesClassifierTester.INSTANCES_COMBINATIONS_FILE_PATH)
    print('Please, assign values to features columns before proceeding.')

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()