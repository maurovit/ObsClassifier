import itertools
import logging
import sys
import csv
import os

COMBINATIONS_ROLES_PREFIX=[['S','O'],
                           ['CS','O'],
                           ['CS','O','CO'],
                           ['S','O','CS','CO']
                           ]

def get_roles_predictions_list(data, predictions, labels):
    i = 0
    classNames = []
    predictionResults = dict()

    for row, column in data.iterrows():
        classNames.append(row[1])

    for pred in predictions:
        class_id = pred['class_ids'][0]
        probability = pred['probabilities'][class_id]
        if (class_id != 4):
            predictionResults.update({classNames[i]: (labels[class_id], probability * 100)})
        i = i + 1

    return predictionResults

def get_instances_predictions_list(data, predictions, labels):
    i = 0
    classNames = []
    predictionResults = dict()

    for row, column in data.iterrows():
        classNames.append(row)

    for pred in predictions:
        class_id = pred['class_ids'][0]
        probability = pred['probabilities'][class_id]
        predictionResults.update({classNames[i]: (labels[class_id], probability * 100)})
        i = i + 1

    return predictionResults

def roles_permutation(predictions_list):
    triplets_list = list(itertools.permutations(predictions_list, 3))
    pairs_list = list(itertools.permutations(predictions_list, 2))
    quadruplets_list=get_quadruplets_list(predictions_list,pairs_list)
    return (quadruplets_list, triplets_list, pairs_list)

def get_quadruplets_list(predictions_list,pairs_list):
    abs_permutation_list = []
    con_permutation_list = []
    for item in pairs_list:
        roleOne = predictions_list[item[0]][0]
        roleTwo = predictions_list[item[1]][0]
        if (roleOne == 'Subject' and roleTwo == 'Observer')or(roleOne == 'Observer' and roleTwo == 'Subject') :
            abs_permutation_list.append(item)
        elif roleOne == 'Subject' and roleTwo == 'Subject':
            abs_permutation_list.append(item)
        elif roleOne == 'Observer' and roleTwo == 'Observer':
            abs_permutation_list.append(item)
        elif(roleOne == 'ConcreteSubject' and roleTwo == 'ConcreteObserver')or(roleOne == 'ConcreteObserver' and roleTwo == 'ConcreteSubject') :
            con_permutation_list.append(item)
        elif roleOne == 'ConcreteSubject' and roleTwo == 'ConcreteSubject':
            con_permutation_list.append(item)
        elif roleOne == 'ConcreteObserver' and roleTwo == 'ConcreteObserver':
            con_permutation_list.append(item)

    quadruplets_temp_list = list(itertools.product(abs_permutation_list, con_permutation_list))
    quadruplets_list=[]

    for quadruplets in quadruplets_temp_list:
        row=[]
        for pair in quadruplets:
            row.append(pair[0])
            row.append(pair[1])
        quadruplets_list.append(tuple(row))

    return quadruplets_list

def filter_pairs_list(prediction_list, pairs_list):
    abs_abs_pairs = []
    con_abs_pairs = []
    for item in pairs_list:
        roleOne = prediction_list[item[0]][0]
        roleTwo = prediction_list[item[1]][0]

        if (roleOne == 'Subject' and roleTwo == 'Observer')or(roleOne == 'Observer' and roleTwo == 'Subject'):
            abs_abs_pairs.append(item)
        elif roleOne == 'Subject' and roleTwo == 'Subject':
            abs_abs_pairs.append(item)
        elif roleOne == 'Observer' and roleTwo == 'Observer':
            abs_abs_pairs.append(item)
        elif roleOne == 'ConcreteSubject' and roleTwo == 'Observer':
            con_abs_pairs.append(item)
        elif roleOne == 'ConcreteSubject' and roleTwo == 'Subject':
            con_abs_pairs.append(item)
        elif roleOne == 'ConcreteObserver' and roleTwo == 'Subject':
            con_abs_pairs.append(item)
        elif roleOne == 'ConcreteObserver' and roleTwo == 'Observer':
            con_abs_pairs.append(item)
    return (abs_abs_pairs, con_abs_pairs)

def filter_triplets_list(prediction_list, triplets_list):
    index=0
    for item in triplets_list:
        roleOne = prediction_list[item[0]][0]
        roleTwo = prediction_list[item[1]][0]
        roleThree = prediction_list[item[2]][0]

        if roleOne == roleTwo and roleTwo == roleThree:
            del triplets_list[index]
        elif roleTwo == 'ConcreteSubject' or roleTwo == 'ConcreteObserver':
            del triplets_list[index]
        elif roleOne == 'Subject' or roleOne == 'Observer':
            del triplets_list[index]
        elif roleThree == 'Subject' or roleThree == 'Observer':
            del triplets_list[index]
        index+=1
    return triplets_list


def get_logger(format,name):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=format)
    logger=logging.getLogger(name=name)
    return logger

def log_combinations_on_file(path,header,combinations):
    with open(path, "w") as fp:
        writer = csv.writer(fp, delimiter=";", dialect="excel", lineterminator="\n")
        writer.writerow(header)
        combinations_index=0
        for classes_set in combinations:
            for combination in classes_set:
                row = ''
                roles_index = 0
                for cl in combination:
                    if roles_index < len(combination)-1:
                        row = row + COMBINATIONS_ROLES_PREFIX[combinations_index][roles_index] + '-' + cl + ','
                    else:
                        row = row + COMBINATIONS_ROLES_PREFIX[combinations_index][roles_index] + '-' + cl
                    roles_index += 1
                writer.writerow([row])
            combinations_index+=1

def log_predictions_on_file(root_directory,path,header,predictions):
    if not os.path.exists(root_directory):
        os.makedirs(root_directory)
    with open(path,"w") as fp:
        writer=csv.writer(fp,delimiter=";", dialect="excel", lineterminator="\n")
        writer.writerow(header)
        for key in predictions:
            writer.writerow([key,predictions[key][0],"%.2f" % round(predictions[key][1],2)])