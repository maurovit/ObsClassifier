import abc
import pandas as pd

class AbstractClassifier(object,metaclass=abc.ABCMeta):

    def __init__(self,columnsName,rolesName):
        self.columnsName = columnsName
        self.rolesName = rolesName
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