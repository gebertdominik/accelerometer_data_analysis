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

features_all = np.loadtxt('mgr/data/resources/all/Features_all.csv', delimiter=",")

dummy_cls_all = DummyClassifier()
k_neighbors_cls_all = KNeighborsClassifier()
decision_tree_cls_all = DecisionTreeClassifier()
random_forest_cls_all = RandomForestClassifier()
mlp_cls_all = MLPClassifier()
gaussian_nb_cls_all = GaussianNB()

print('Test Classifiers on ALL learned with data collected by ALL')
print('Dummy Classifier        ', sig.test_and_learn_classifier(dummy_cls_all, features_all))
print('K-Neighbors Classifier  ', sig.test_and_learn_classifier(k_neighbors_cls_all, features_all))
print('Decision Tree Classifier', sig.test_and_learn_classifier(decision_tree_cls_all, features_all))
print('Random Forest Classifier', sig.test_and_learn_classifier(random_forest_cls_all, features_all))
print('MLP Classifier          ', sig.test_and_learn_classifier(mlp_cls_all, features_all))
print('GaussianNB              ', sig.test_and_learn_classifier(gaussian_nb_cls_all, features_all))
