import mgr.data.readrawdata as rd
import mgr.calc.signal as sig
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import csv

""" PERSON_1 - prepare data - start """
STANDING_PERSON_1 = rd.read('mgr/data/resources/ok/standing.csv')
sig.denoise(STANDING_PERSON_1)
STANDING_PERSON_1['magnitude'] = sig.magnitude(STANDING_PERSON_1)

WALKING_PERSON_1 = rd.read('mgr/data/resources/ok/walking.csv')
sig.denoise(WALKING_PERSON_1)
WALKING_PERSON_1['magnitude'] = sig.magnitude(WALKING_PERSON_1)

DOWNSTAIRS_PERSON_1 = rd.read('mgr/data/resources/ok/downstairs.csv')
sig.denoise(DOWNSTAIRS_PERSON_1)
DOWNSTAIRS_PERSON_1['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_1)

UPSTAIRS_PERSON_1 = rd.read('mgr/data/resources/ok/upstairs.csv')
sig.denoise(UPSTAIRS_PERSON_1)
UPSTAIRS_PERSON_1['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_1)

RUNNING_PERSON_1 = rd.read('mgr/data/resources/ok/running.csv')
sig.denoise(RUNNING_PERSON_1)
RUNNING_PERSON_1['magnitude'] = sig.magnitude(RUNNING_PERSON_1)

""" PERSON_1 - prepare data - stop """

""" ... other persons"""

activities_person_1 = [STANDING_PERSON_1, WALKING_PERSON_1, DOWNSTAIRS_PERSON_1, UPSTAIRS_PERSON_1, RUNNING_PERSON_1]
output_file_path_person_1 = 'mgr/data/resources/Features_person_1.csv'

with open(output_file_path_person_1, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_1)):
        for f in sig.extract_features(activities_person_1[i]):
            rows.writerow([i] + f)

features_person_1 = np.loadtxt('mgr/data/resources/Features_person_1.csv', delimiter=",")

dummy_cls_person_1 = DummyClassifier()
k_neighbors_cls_person_1 = KNeighborsClassifier()
decision_tree_cls_person_1 = DecisionTreeClassifier()
random_forest_cls_person_1 = RandomForestClassifier()
mlp_cls_person_1 = MLPClassifier()
gaussian_nb_cls_person_1 = GaussianNB()

print('Test Classifiers on PERSON_1 learned with data collected by PERSON_1')
print('Dummy Classifier        ', sig.test_classifier(dummy_cls_person_1, features_person_1))
print('K-Neighbors Classifier  ', sig.test_classifier(k_neighbors_cls_person_1, features_person_1))
print('Decision Tree Classifier', sig.test_classifier(decision_tree_cls_person_1, features_person_1))
print('Random Forest Classifier', sig.test_classifier(random_forest_cls_person_1, features_person_1))
print('MLP Classifier          ', sig.test_classifier(mlp_cls_person_1, features_person_1))
print('GaussianNB              ', sig.test_classifier(gaussian_nb_cls_person_1, features_person_1))