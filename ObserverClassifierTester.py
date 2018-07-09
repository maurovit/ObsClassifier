import DatasetUtilities
import pandas as pd
import os
from RolesClassifier import RolesClassifier
from InstancesClassifier import InstancesClassifier

ROLES_DATASET_PATH    =  "datasets/obs_roles_dataset.csv"
ROLES_FEATURE_COLUMNS = ['CollectionVariables','AddListenerMethod','RemoveListenerMethod','ClassDeclarationKeyword',
                         'MethodsDeclarationKeyword','ClassType','ScanCollectionsMethod','SCMCallsAbsMethod','HasSuperclass',
                         'ImplementsInterfaces','ChangeState','AfterChangeStateIterateOverList','Role']
ROLES_LABELS          = ['Subject', 'Observer', 'ConcreteSubject', 'ConcreteObserver','None']

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

def main():
    '''
    rolesClassifier=RolesClassifier(ROLES_FEATURE_COLUMNS,ROLES_LABELS)
    rolesClassifier.initFeatureColumns()
    rolesClassifier.initClassifier()
    rolesClassifier.loadDataset(ROLES_DATASET_PATH,0,';')
    rolesClassifier.suffleDataset()
    rolesFolders=DatasetUtilities.k_folders(rolesClassifier.getDataset(),ROLES_FEATURE_COLUMNS[0],RolesClassifier.FOLDERS_NUMBER)
    y_name=ROLES_FEATURE_COLUMNS[len(ROLES_FEATURE_COLUMNS)-1]

    for i in range(RolesClassifier.FOLDERS_NUMBER):
        testSetIndex=i+1
        print("\nTEST SET - ","FOLDER",testSetIndex)
        traingSetFolders=[]
        print("TRAINING FOLDERS - ",end='')

        for j in range(i):
            print("FOLDER",(j+1)," ",end='')
            traingSetFolders.append(rolesFolders[DatasetUtilities.KEYS_PREFIX + str(j+1)])
        for j in range(i+1,RolesClassifier.FOLDERS_NUMBER):
            print("FOLDER",(j+1)," ",end='')
            traingSetFolders.append(rolesFolders[DatasetUtilities.KEYS_PREFIX + str(j+1)])

        trainingSet=pd.concat(traingSetFolders)
        testSet=rolesFolders[DatasetUtilities.KEYS_PREFIX + str(testSetIndex)]
        rolesClassifier.setTrainingSet(trainingSet)
        rolesClassifier.setTestSet(testSet)
        print("\nPHASE "+str(i+1)+" - ","Training....")
        rolesClassifier.train(y_name,ROLES_TRAIN_BATCH_SIZE,ROLES_TRAINING_STEPS)
        print("PHASE "+str(i+1)+" - ","Evaluating....")
        rolesClassifier.evaluate(y_name,ROLES_EVALUATE_BATCH_SIZE)
        print("RESULT - ","Test set accuracy: {accuracy:0.3f}\n".format(**rolesClassifier.getEvaluationResult()))

    print("FINAL RESULT - Accuracy: %.3f"%rolesClassifier.getAvgAccuracy())'''

    instancesClassifier = InstancesClassifier(INSTANCES_FEATURE_COLUMNS, INSTANCE_LABELS)
    instancesClassifier.initFeatureColumns()
    instancesClassifier.initClassifier()
    instancesClassifier.loadDataset(INSTANCES_DATASET_PATH, 0, ';')
    instancesClassifier.suffleDataset()

    rolesFolders = DatasetUtilities.k_folders(instancesClassifier.getDataset(), INSTANCES_FEATURE_COLUMNS[0], InstancesClassifier.FOLDERS_NUMBER)
    y_name = INSTANCES_FEATURE_COLUMNS[len(INSTANCES_FEATURE_COLUMNS) - 1]

    for i in range(InstancesClassifier.FOLDERS_NUMBER):
        testSetIndex = i + 1
        print("\nTEST SET - ", "FOLDER", testSetIndex)
        traingSetFolders = []
        print("TRAINING FOLDERS - ", end='')

        for j in range(i):
            print("FOLDER", (j + 1), " ", end='')
            traingSetFolders.append(rolesFolders[DatasetUtilities.KEYS_PREFIX + str(j + 1)])
        for j in range(i + 1, InstancesClassifier.FOLDERS_NUMBER):
            print("FOLDER", (j + 1), " ", end='')
            traingSetFolders.append(rolesFolders[DatasetUtilities.KEYS_PREFIX + str(j + 1)])

        trainingSet = pd.concat(traingSetFolders)
        testSet = rolesFolders[DatasetUtilities.KEYS_PREFIX + str(testSetIndex)]
        instancesClassifier.setTrainingSet(trainingSet)
        instancesClassifier.setTestSet(testSet)
        print("\nPHASE " + str(i + 1) + " - ", "Training....")
        instancesClassifier.train(y_name, INSTANCES_TRAIN_BATCH_SIZE, INSTANCES_TRAINING_STEPS)
        print("PHASE " + str(i + 1) + " - ", "Evaluating....")
        instancesClassifier.evaluate(y_name, INSTANCES_EVALUATE_BATCH_SIZE)
        print("RESULT - ", "Test set accuracy: {accuracy:0.3f}\n".format(**instancesClassifier.getEvaluationResult()))

    print("FINAL RESULT - Accuracy: %.3f" % instancesClassifier.getAvgAccuracy())


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()