from AbstractClassifier import AbstractClassifier
import tensorflow as tf
import DatasetUtilities

class InstancesClassifier(AbstractClassifier):

    def __init__(self,columnsName, rolesName, foldersNumber):
        super().__init__(columnsName, rolesName, foldersNumber)
        self.evaluationResult = None
        self.avgAccurcy = 0.0
        self.trainsNumber = 0

    def getEvaluationResult(self):
        return self.evaluationResult

    def initFeatureColumns(self):
        self.featureColumns = []

        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[0], num_buckets=3))
        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[1], num_buckets=3))
        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[2], num_buckets=4))
        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[3], num_buckets=5))
        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[4], num_buckets=5))
        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[5], num_buckets=4))
        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[6], num_buckets=6))
        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[7], num_buckets=3))
        self.featureColumns.append(tf.feature_column.categorical_column_with_identity(key=self.columnsName[8], num_buckets=5))

    def initClassifier(self):
        self.classifier=tf.estimator.LinearClassifier(feature_columns=self.featureColumns,
                                                      optimizer=tf.train.FtrlOptimizer(learning_rate=0.001),
                                                      loss_reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                                                      n_classes=len(self.rolesName))


    def train(self,label_col_name,batch_size,training_steps):
        train_x, train_y=self.trainingSet, self.trainingSet.copy().pop(label_col_name)
        self.classifier.train(input_fn=lambda:DatasetUtilities.train_input_fn(train_x,train_y,batch_size),steps=training_steps)
        self.trainsNumber+=1

    def evaluate(self,label_col_name,batch_size):
        test_x, test_y=self.testSet, self.testSet.copy().pop(label_col_name)
        self.evaluationResult=self.classifier.evaluate(input_fn=lambda:DatasetUtilities.eval_input_fn(test_x,test_y,batch_size))
        self.avgAccurcy=self.avgAccurcy+self.evaluationResult['accuracy']

    def getAvgAccuracy(self):
        return self.avgAccurcy / self.trainsNumber