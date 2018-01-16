import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.signal import lfilter, butter
from scipy.signal import lfilter_zi

import mgr.data.readrawdata as rd
import mgr.visualization.visualize as vis
import mgr.calc.signal as sig
import numpy as np
from prepare_data import *

features_person_1 = np.loadtxt(features_file_person_1, delimiter=",")
features_person_2 = np.loadtxt(features_file_person_2, delimiter=",")
features_person_3 = np.loadtxt(features_file_person_3, delimiter=",")
features_person_4 = np.loadtxt(features_file_person_4, delimiter=",")

features_all = np.concatenate([features_person_1, features_person_2, features_person_3, features_person_4])
features_person_1_2_3 = np.concatenate([features_person_1, features_person_2, features_person_3])
features_person_1_2_4 = np.concatenate([features_person_1, features_person_2, features_person_4])
features_person_1_3_4 = np.concatenate([features_person_1, features_person_3, features_person_4])
features_person_2_3_4 = np.concatenate([features_person_2, features_person_3, features_person_4])

loops = 20
#for i in range(0, loops):
dummy_cls = DummyClassifier()
k_neighbors_cls= KNeighborsClassifier()
decision_tree_cls = DecisionTreeClassifier()
random_forest_cls = RandomForestClassifier()
mlp_cls = MLPClassifier()
gaussian_nb_cls = GaussianNB()
activity_features = features_person_1_2_3[:, 1:]
activity_markers = features_person_1_2_3[:, 0]
af_train, af_test, am_train, am_test = train_test_split(activity_features, activity_markers, test_size=.25)
dummy_cls.fit(af_train, am_train)
k_neighbors_cls.fit(af_train, am_train)
decision_tree_cls.fit(af_train, am_train)
random_forest_cls.fit(af_train, am_train)
mlp_cls.fit(af_train, am_train)
gaussian_nb_cls.fit(af_train, am_train)
print('Test Classifiers on PERSON_1 learned with 2,3,4')
#print('Dummy Classifier        ', sig.test_classifier(dummy_cls, features_person_4))
print('K-Neighbors Classifier  ', sig.test_and_learn_knn(features_person_1_2_3, features_person_4))
#print('Decision Tree Classifier', sig.test_classifier(decision_tree_cls, features_person_4))
#print('Random Forest Classifier', sig.test_classifier(random_forest_cls, features_person_4))
#print('MLP Classifier          ', sig.test_classifier(mlp_cls, features_person_4))
#print('GaussianNB              ', sig.test_classifier(gaussian_nb_cls, features_person_4))


