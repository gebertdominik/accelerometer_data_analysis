
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

print('\nTest Classifiers - learn: 1,2,3 test: 4')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_person_1_2_3, features_person_4))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_person_1_2_3, features_person_4))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_person_1_2_3, features_person_4))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_person_1_2_3, features_person_4))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_person_1_2_3, features_person_4))

print('\nTest Classifiers - learn: 1,2,4 test: 3')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_person_1_2_4, features_person_3))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_person_1_2_4, features_person_3))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_person_1_2_4, features_person_3))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_person_1_2_4, features_person_3))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_person_1_2_4, features_person_3))

print('\nTest Classifiers - learn: 1,3,4 test: 2')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_person_1_3_4, features_person_2))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_person_1_3_4, features_person_2))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_person_1_3_4, features_person_2))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_person_1_3_4, features_person_2))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_person_1_3_4, features_person_2))

print('\nTest Classifiers - learn: 2,3,4 test: 1')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_person_2_3_4, features_person_1))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_person_2_3_4, features_person_1))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_person_2_3_4, features_person_1))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_person_2_3_4, features_person_1))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_person_2_3_4, features_person_1))

