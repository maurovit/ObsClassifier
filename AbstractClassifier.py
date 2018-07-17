import abc
import pandas as pd
import DatasetUtilities
import logging
import sys

class AbstractClassifier(object,metaclass=abc.ABCMeta):

    def __init__(self,columnsName,rolesName,foldersNumber):
        self.columnsName = columnsName
        self.rolesName = rolesName
        self.foldersNumber=foldersNumber
        self.featureColumns = None
        self.classifier = None
        self.dataSet = None
        self.trainingSet = None
        self.testSet = None

    def getDataset(self):
        return self.dataSet

    def getTrainingSet(self):
        return self.trainingSet

    def setTrainingSet(self,newTrainingSet):
        self.trainingSet=newTrainingSet

    def setTestSet(self,newTestSet):
        self.testSet=newTestSet

    def loadDataset(self,trainPath,header,delimitier):
        self.dataSet=pd.read_csv(trainPath,names=self.columnsName,header=header,delimiter=delimitier)

    def suffleDataset(self):
        self.dataSet=self.dataSet.sample(frac=1)

    def kFoldersTrainAndEvaluation(self,train_batch_size,train_steps,evaluation_batch_size,useLogger):

        rolesFolders = DatasetUtilities.k_folders(self.dataSet, self.columnsName[0], self.foldersNumber)
        y_name = self.columnsName[len(self.columnsName) - 1]

        logging.basicConfig(stream=sys.stdout,level=logging.DEBUG,format='%(name)s - %(message)s')
        logger=logging.getLogger(self.__class__.__name__)
        logger.propagate=useLogger
        logging.disable(logging.INFO)

        for i in range(self.foldersNumber):
            testSetIndex = i + 1
            logger.warning("TEST SET - FOLDER "+str(testSetIndex))
            trainingSetFolders = []
            logger.warning("TRAINING FOLDERS ...")
            for j in range(i):
                logger.warning("FOLDER "+str(j + 1)+" ")
                trainingSetFolders.append(rolesFolders[DatasetUtilities.KEYS_PREFIX + str(j + 1)])
            for j in range(i + 1, self.foldersNumber):
                logger.warning("FOLDER "+str(j + 1)+" ")
                trainingSetFolders.append(rolesFolders[DatasetUtilities.KEYS_PREFIX + str(j + 1)])

            trainingSet = pd.concat(trainingSetFolders)
            testSet = rolesFolders[DatasetUtilities.KEYS_PREFIX + str(testSetIndex)]
            self.setTrainingSet(trainingSet)
            self.setTestSet(testSet)
            logger.warning("PHASE " + str(i + 1) + " - Training....")
            self.train(y_name, train_batch_size, train_steps)
            logger.warning("PHASE " + str(i + 1) + " - Evaluating....")
            self.evaluate(y_name, evaluation_batch_size)
            logger.warning("PHASE " + str(i + 1) +" RESULT - Test set accuracy: {accuracy:0.3f}\n".format(**self.getEvaluationResult()))

        logger.warning("FINAL RESULT - Avg Accuracy: %.3f"%self.getAvgAccuracy())

    @abc.abstractmethod
    def initFeatureColumns(self):
        return

    @abc.abstractmethod
    def initClassifier(self):
        return

    @abc.abstractmethod
    def train(self,labels_name,batch_size,training_steps):
        return

    @abc.abstractmethod
    def evaluate(self,labels_name,batch_size):
        return

    @abc.abstractmethod
    def predict(self,data_path,header,delimiter,batch_size):
        return