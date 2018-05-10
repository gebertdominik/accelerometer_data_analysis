import mgr.data.readrawdata as rd
import mgr.calc.signal as sig
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv

""" PERSON_1 - prepare data - start """
STANDING_PERSON_1 = rd.read('mgr/data/resources/person_1/standing.csv')
STANDING_PERSON_1['magnitude'] = sig.magnitude(STANDING_PERSON_1)

WALKING_PERSON_1 = rd.read('mgr/data/resources/person_1/walking.csv')
WALKING_PERSON_1['magnitude'] = sig.magnitude(WALKING_PERSON_1)

DOWNSTAIRS_PERSON_1 = rd.read('mgr/data/resources/person_1/downstairs.csv')
DOWNSTAIRS_PERSON_1['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_1)

UPSTAIRS_PERSON_1 = rd.read('mgr/data/resources/person_1/upstairs.csv')
UPSTAIRS_PERSON_1['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_1)

RUNNING_PERSON_1 = rd.read('mgr/data/resources/person_1/running.csv')
RUNNING_PERSON_1['magnitude'] = sig.magnitude(RUNNING_PERSON_1)

""" PERSON_1 - prepare data - stop """

activities_person_1 = [STANDING_PERSON_1, WALKING_PERSON_1, DOWNSTAIRS_PERSON_1, UPSTAIRS_PERSON_1, RUNNING_PERSON_1]
output_file_path_person_1 = 'mgr/data/features/person_1/Features.csv'

with open(output_file_path_person_1, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_1)):
        for f in sig.extract_features(activities_person_1[i]):
            rows.writerow([i] + f)

features_person_1 = np.loadtxt('mgr/data/features/person_1/Features.csv', delimiter=",")

print('\nTest Classifiers on PERSON_1 learned with data collected by PERSON_1')
print('K-Neighbors Classifier  ', sig.test_knn_cls_one_set(features_person_1))
print('Decision Tree Classifier', sig.test_decision_tree_cls_one_set(features_person_1))
print('Random Forest Classifier', sig.test_random_forest_cls_one_set(features_person_1))
print('MLP Classifier          ', sig.test_mlp_cls_one_set(features_person_1))
print('GaussianNB              ', sig.test_gaussian_nb_cls_one_set(features_person_1))


""" PERSON_2 - prepare data - start """
STANDING_PERSON_2 = rd.read('mgr/data/resources/person_2/standing.csv')
STANDING_PERSON_2['magnitude'] = sig.magnitude(STANDING_PERSON_2)

WALKING_PERSON_2 = rd.read('mgr/data/resources/person_2/walking.csv')
WALKING_PERSON_2['magnitude'] = sig.magnitude(WALKING_PERSON_2)

DOWNSTAIRS_PERSON_2 = rd.read('mgr/data/resources/person_2/downstairs.csv')
DOWNSTAIRS_PERSON_2['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_2)

UPSTAIRS_PERSON_2 = rd.read('mgr/data/resources/person_2/upstairs.csv')
UPSTAIRS_PERSON_2['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_2)

RUNNING_PERSON_2 = rd.read('mgr/data/resources/person_2/running.csv')
RUNNING_PERSON_2['magnitude'] = sig.magnitude(RUNNING_PERSON_2)

""" PERSON_2 - prepare data - stop """

activities_person_2 = [STANDING_PERSON_2, WALKING_PERSON_2, DOWNSTAIRS_PERSON_2, UPSTAIRS_PERSON_2, RUNNING_PERSON_2]
output_file_path_person_2 = 'mgr/data/features/person_2/Features.csv'

with open(output_file_path_person_2, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_2)):
        for f in sig.extract_features(activities_person_2[i]):
            rows.writerow([i] + f)

features_person_2 = np.loadtxt('mgr/data/features/person_2/Features.csv', delimiter=",")

print('\nTest Classifiers on PERSON_2 learned with data collected by PERSON_2')
print('K-Neighbors Classifier  ', sig.test_knn_cls_one_set(features_person_2))
print('Decision Tree Classifier', sig.test_decision_tree_cls_one_set(features_person_2))
print('Random Forest Classifier', sig.test_random_forest_cls_one_set(features_person_2))
print('MLP Classifier          ', sig.test_mlp_cls_one_set(features_person_2))
print('GaussianNB              ', sig.test_gaussian_nb_cls_one_set(features_person_2))


""" PERSON_3 - prepare data - start """
STANDING_PERSON_3 = rd.read('mgr/data/resources/person_3/standing.csv')
STANDING_PERSON_3['magnitude'] = sig.magnitude(STANDING_PERSON_3)

WALKING_PERSON_3 = rd.read('mgr/data/resources/person_3/walking.csv')
WALKING_PERSON_3['magnitude'] = sig.magnitude(WALKING_PERSON_3)

DOWNSTAIRS_PERSON_3 = rd.read('mgr/data/resources/person_3/downstairs.csv')
DOWNSTAIRS_PERSON_3['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_3)

UPSTAIRS_PERSON_3 = rd.read('mgr/data/resources/person_3/upstairs.csv')
UPSTAIRS_PERSON_3['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_3)

RUNNING_PERSON_3 = rd.read('mgr/data/resources/person_3/running.csv')
RUNNING_PERSON_3['magnitude'] = sig.magnitude(RUNNING_PERSON_3)

""" PERSON_3 - prepare data - stop """

activities_person_3 = [STANDING_PERSON_3, WALKING_PERSON_3, DOWNSTAIRS_PERSON_3, UPSTAIRS_PERSON_3, RUNNING_PERSON_3]
output_file_path_person_3 = 'mgr/data/features/person_3/Features.csv'

with open(output_file_path_person_3, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_3)):
        for f in sig.extract_features(activities_person_3[i]):
            rows.writerow([i] + f)

features_person_3 = np.loadtxt('mgr/data/features/person_3/Features.csv', delimiter=",")

print('\nTest Classifiers on PERSON_3 learned with data collected by PERSON_3')
print('K-Neighbors Classifier  ', sig.test_knn_cls_one_set(features_person_3))
print('Decision Tree Classifier', sig.test_decision_tree_cls_one_set(features_person_3))
print('Random Forest Classifier', sig.test_random_forest_cls_one_set(features_person_3))
print('MLP Classifier          ', sig.test_mlp_cls_one_set(features_person_3))
print('GaussianNB              ', sig.test_gaussian_nb_cls_one_set(features_person_3))


""" PERSON_4 - prepare data - start """
STANDING_PERSON_4 = rd.read('mgr/data/resources/person_4/standing.csv')
STANDING_PERSON_4['magnitude'] = sig.magnitude(STANDING_PERSON_4)

WALKING_PERSON_4 = rd.read('mgr/data/resources/person_4/walking.csv')
WALKING_PERSON_4['magnitude'] = sig.magnitude(WALKING_PERSON_4)

DOWNSTAIRS_PERSON_4 = rd.read('mgr/data/resources/person_4/downstairs.csv')
DOWNSTAIRS_PERSON_4['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_4)

UPSTAIRS_PERSON_4 = rd.read('mgr/data/resources/person_4/upstairs.csv')
UPSTAIRS_PERSON_4['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_4)

RUNNING_PERSON_4 = rd.read('mgr/data/resources/person_4/running.csv')
RUNNING_PERSON_4['magnitude'] = sig.magnitude(RUNNING_PERSON_4)

""" PERSON_4 - prepare data - stop """

activities_person_4 = [STANDING_PERSON_4, WALKING_PERSON_4, DOWNSTAIRS_PERSON_4, UPSTAIRS_PERSON_4, RUNNING_PERSON_4]
output_file_path_person_4 = 'mgr/data/features/person_4/Features.csv'

with open(output_file_path_person_4, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_4)):
        for f in sig.extract_features(activities_person_4[i]):
            rows.writerow([i] + f)

features_person_4 = np.loadtxt('mgr/data/features/person_4/Features.csv', delimiter=",")


print('\nTest Classifiers on PERSON_4 learned with data collected by PERSON_4')
print('K-Neighbors Classifier  ', sig.test_knn_cls_one_set(features_person_4))
print('Decision Tree Classifier', sig.test_decision_tree_cls_one_set(features_person_4))
print('Random Forest Classifier', sig.test_random_forest_cls_one_set(features_person_4))
print('MLP Classifier          ', sig.test_mlp_cls_one_set(features_person_4))
print('GaussianNB              ', sig.test_gaussian_nb_cls_one_set(features_person_4))

features_all = np.concatenate([features_person_1, features_person_2, features_person_3, features_person_4])

k_neighbors_cls_all = KNeighborsClassifier()
decision_tree_cls_all = DecisionTreeClassifier()
random_forest_cls_all = RandomForestClassifier()
mlp_cls_all = MLPClassifier()
gaussian_nb_cls_all = GaussianNB()


print('\nTest Classifiers on PERSON_1 learned with all data')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_all, features_person_1))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_all, features_person_1))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_all, features_person_1))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_all, features_person_1))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_all, features_person_1))

print('\nTest Classifiers on PERSON_2 learned with all data')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_all, features_person_2))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_all, features_person_2))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_all, features_person_2))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_all, features_person_2))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_all, features_person_2))

print('\nTest Classifiers on PERSON_3 learned with all data')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_all, features_person_3))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_all, features_person_3))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_all, features_person_3))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_all, features_person_3))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_all, features_person_3))


print('\nTest Classifiers on PERSON_4 learned with all data')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_all, features_person_4))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_all, features_person_4))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_all, features_person_4))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_all, features_person_4))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_all, features_person_4))
