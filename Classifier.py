from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,help='number of training steps')

def main(argv):
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    feature_columns = [
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[0], num_buckets=3)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[1], num_buckets=3)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[2], num_buckets=3)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[3], num_buckets=4)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[4], num_buckets=4)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[5], num_buckets=4)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[6], num_buckets=3)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[7], num_buckets=3)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[8], num_buckets=3)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[9], num_buckets=3)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[10], num_buckets=3)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key=iris_data.CSV_COLUMN_NAMES[11], num_buckets=3))
    ]

    '''
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[13, 13, 13],
        # The model must choose between 3 classes.
        n_classes=5)
        '''
    classifier=tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                             optimizer=tf.train.FtrlOptimizer(learning_rate=0.01),
                                             loss_reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                                             n_classes=5)

    # Train the Model.
    classifier.train(input_fn=lambda:iris_data.train_input_fn(train_x, train_y,10),steps=1000)

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,10))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    '''
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(input_fn=lambda:iris_data.eval_input_fn(predict_x,labels=None,batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(iris_data.SPECIES[class_id],100 * probability, expec))
    '''

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)