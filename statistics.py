import mgr.data.readrawdata as rd
import mgr.calc.signal as sig
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv

""" PERSON_1 - prepare data - start """
STANDING_PERSON_1 = rd.read('mgr/data/resources/person_1/standing.csv')
sig.denoise(STANDING_PERSON_1)
STANDING_PERSON_1['magnitude'] = sig.magnitude(STANDING_PERSON_1)

WALKING_PERSON_1 = rd.read('mgr/data/resources/person_1/walking.csv')
sig.denoise(WALKING_PERSON_1)
WALKING_PERSON_1['magnitude'] = sig.magnitude(WALKING_PERSON_1)

DOWNSTAIRS_PERSON_1 = rd.read('mgr/data/resources/person_1/downstairs.csv')
sig.denoise(DOWNSTAIRS_PERSON_1)
DOWNSTAIRS_PERSON_1['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_1)

UPSTAIRS_PERSON_1 = rd.read('mgr/data/resources/person_1/upstairs.csv')
sig.denoise(UPSTAIRS_PERSON_1)
UPSTAIRS_PERSON_1['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_1)

RUNNING_PERSON_1 = rd.read('mgr/data/resources/person_1/running.csv')
sig.denoise(RUNNING_PERSON_1)
RUNNING_PERSON_1['magnitude'] = sig.magnitude(RUNNING_PERSON_1)

""" PERSON_1 - prepare data - stop """

activities_person_1 = [STANDING_PERSON_1, WALKING_PERSON_1, DOWNSTAIRS_PERSON_1, UPSTAIRS_PERSON_1, RUNNING_PERSON_1]
output_file_path_person_1 = 'mgr/data/resources/person_1/Features.csv'

with open(output_file_path_person_1, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_1)):
        for f in sig.extract_features(activities_person_1[i]):
            rows.writerow([i] + f)

features_person_1 = np.loadtxt('mgr/data/resources/person_1/Features.csv', delimiter=",")

dummy_cls_person_1 = DummyClassifier()
k_neighbors_cls_person_1 = KNeighborsClassifier()
decision_tree_cls_person_1 = DecisionTreeClassifier()
random_forest_cls_person_1 = RandomForestClassifier()
mlp_cls_person_1 = MLPClassifier(max_iter=1000)
gaussian_nb_cls_person_1 = GaussianNB()

print('Test Classifiers on PERSON_1 learned with data collected by PERSON_1')
print('Dummy Classifier        ', sig.test_and_learn_classifier(dummy_cls_person_1, features_person_1))
print('K-Neighbors Classifier  ', sig.test_and_learn_classifier(k_neighbors_cls_person_1, features_person_1))
print('Decision Tree Classifier', sig.test_and_learn_classifier(decision_tree_cls_person_1, features_person_1))
print('Random Forest Classifier', sig.test_and_learn_classifier(random_forest_cls_person_1, features_person_1))
print('MLP Classifier          ', sig.test_and_learn_classifier(mlp_cls_person_1, features_person_1))
print('GaussianNB              ', sig.test_and_learn_classifier(gaussian_nb_cls_person_1, features_person_1))


""" PERSON_2 - prepare data - start """
STANDING_PERSON_2 = rd.read('mgr/data/resources/person_2/standing.csv')
sig.denoise(STANDING_PERSON_2)
STANDING_PERSON_2['magnitude'] = sig.magnitude(STANDING_PERSON_2)

WALKING_PERSON_2 = rd.read('mgr/data/resources/person_2/walking.csv')
sig.denoise(WALKING_PERSON_2)
WALKING_PERSON_2['magnitude'] = sig.magnitude(WALKING_PERSON_2)

DOWNSTAIRS_PERSON_2 = rd.read('mgr/data/resources/person_2/downstairs.csv')
sig.denoise(DOWNSTAIRS_PERSON_2)
DOWNSTAIRS_PERSON_2['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_2)

UPSTAIRS_PERSON_2 = rd.read('mgr/data/resources/person_2/upstairs.csv')
sig.denoise(UPSTAIRS_PERSON_2)
UPSTAIRS_PERSON_2['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_2)

RUNNING_PERSON_2 = rd.read('mgr/data/resources/person_2/running.csv')
sig.denoise(RUNNING_PERSON_2)
RUNNING_PERSON_2['magnitude'] = sig.magnitude(RUNNING_PERSON_2)

""" PERSON_2 - prepare data - stop """

activities_person_2 = [STANDING_PERSON_2, WALKING_PERSON_2, DOWNSTAIRS_PERSON_2, UPSTAIRS_PERSON_2, RUNNING_PERSON_2]
output_file_path_person_2 = 'mgr/data/resources/person_2/Features.csv'

with open(output_file_path_person_2, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_2)):
        for f in sig.extract_features(activities_person_2[i]):
            rows.writerow([i] + f)

features_person_2 = np.loadtxt('mgr/data/resources/person_2/Features.csv', delimiter=",")

dummy_cls_person_2 = DummyClassifier()
k_neighbors_cls_person_2 = KNeighborsClassifier()
decision_tree_cls_person_2 = DecisionTreeClassifier()
random_forest_cls_person_2 = RandomForestClassifier()
mlp_cls_person_2 = MLPClassifier()
gaussian_nb_cls_person_2 = GaussianNB()

print('Test Classifiers on PERSON_2 learned with data collected by PERSON_2')
print('Dummy Classifier        ', sig.test_and_learn_classifier(dummy_cls_person_2, features_person_2))
print('K-Neighbors Classifier  ', sig.test_and_learn_classifier(k_neighbors_cls_person_2, features_person_2))
print('Decision Tree Classifier', sig.test_and_learn_classifier(decision_tree_cls_person_2, features_person_2))
print('Random Forest Classifier', sig.test_and_learn_classifier(random_forest_cls_person_2, features_person_2))
print('MLP Classifier          ', sig.test_and_learn_classifier(mlp_cls_person_2, features_person_2))
print('GaussianNB              ', sig.test_and_learn_classifier(gaussian_nb_cls_person_2, features_person_2))


""" PERSON_3 - prepare data - start """
STANDING_PERSON_3 = rd.read('mgr/data/resources/person_3/standing.csv')
sig.denoise(STANDING_PERSON_3)
STANDING_PERSON_3['magnitude'] = sig.magnitude(STANDING_PERSON_3)

WALKING_PERSON_3 = rd.read('mgr/data/resources/person_3/walking.csv')
sig.denoise(WALKING_PERSON_3)
WALKING_PERSON_3['magnitude'] = sig.magnitude(WALKING_PERSON_3)

DOWNSTAIRS_PERSON_3 = rd.read('mgr/data/resources/person_3/downstairs.csv')
sig.denoise(DOWNSTAIRS_PERSON_3)
DOWNSTAIRS_PERSON_3['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_3)

UPSTAIRS_PERSON_3 = rd.read('mgr/data/resources/person_3/upstairs.csv')
sig.denoise(UPSTAIRS_PERSON_3)
UPSTAIRS_PERSON_3['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_3)

RUNNING_PERSON_3 = rd.read('mgr/data/resources/person_3/running.csv')
sig.denoise(RUNNING_PERSON_3)
RUNNING_PERSON_3['magnitude'] = sig.magnitude(RUNNING_PERSON_3)

""" PERSON_3 - prepare data - stop """

activities_person_3 = [STANDING_PERSON_3, WALKING_PERSON_3, DOWNSTAIRS_PERSON_3, UPSTAIRS_PERSON_3, RUNNING_PERSON_3]
output_file_path_person_3 = 'mgr/data/resources/person_3/Features.csv'

with open(output_file_path_person_3, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_3)):
        for f in sig.extract_features(activities_person_3[i]):
            rows.writerow([i] + f)

features_person_3 = np.loadtxt('mgr/data/resources/person_3/Features.csv', delimiter=",")

dummy_cls_person_3 = DummyClassifier()
k_neighbors_cls_person_3 = KNeighborsClassifier()
decision_tree_cls_person_3 = DecisionTreeClassifier()
random_forest_cls_person_3 = RandomForestClassifier()
mlp_cls_person_3 = MLPClassifier()
gaussian_nb_cls_person_3 = GaussianNB()

print('Test Classifiers on PERSON_3 learned with data collected by PERSON_3')
print('Dummy Classifier        ', sig.test_and_learn_classifier(dummy_cls_person_3, features_person_3))
print('K-Neighbors Classifier  ', sig.test_and_learn_classifier(k_neighbors_cls_person_3, features_person_3))
print('Decision Tree Classifier', sig.test_and_learn_classifier(decision_tree_cls_person_3, features_person_3))
print('Random Forest Classifier', sig.test_and_learn_classifier(random_forest_cls_person_3, features_person_3))
print('MLP Classifier          ', sig.test_and_learn_classifier(mlp_cls_person_3, features_person_3))
print('GaussianNB              ', sig.test_and_learn_classifier(gaussian_nb_cls_person_3, features_person_3))


""" PERSON_4 - prepare data - start """
STANDING_PERSON_4 = rd.read('mgr/data/resources/person_4/standing.csv')
sig.denoise(STANDING_PERSON_4)
STANDING_PERSON_4['magnitude'] = sig.magnitude(STANDING_PERSON_4)

WALKING_PERSON_4 = rd.read('mgr/data/resources/person_4/walking.csv')
sig.denoise(WALKING_PERSON_4)
WALKING_PERSON_4['magnitude'] = sig.magnitude(WALKING_PERSON_4)

DOWNSTAIRS_PERSON_4 = rd.read('mgr/data/resources/person_4/downstairs.csv')
sig.denoise(DOWNSTAIRS_PERSON_4)
DOWNSTAIRS_PERSON_4['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_4)

UPSTAIRS_PERSON_4 = rd.read('mgr/data/resources/person_4/upstairs.csv')
sig.denoise(UPSTAIRS_PERSON_4)
UPSTAIRS_PERSON_4['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_4)

RUNNING_PERSON_4 = rd.read('mgr/data/resources/person_4/running.csv')
sig.denoise(RUNNING_PERSON_4)
RUNNING_PERSON_4['magnitude'] = sig.magnitude(RUNNING_PERSON_4)

""" PERSON_4 - prepare data - stop """

activities_person_4 = [STANDING_PERSON_4, WALKING_PERSON_4, DOWNSTAIRS_PERSON_4, UPSTAIRS_PERSON_4, RUNNING_PERSON_4]
output_file_path_person_4 = 'mgr/data/resources/person_4/Features.csv'

with open(output_file_path_person_4, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_4)):
        for f in sig.extract_features(activities_person_4[i]):
            rows.writerow([i] + f)

features_person_4 = np.loadtxt('mgr/data/resources/person_4/Features.csv', delimiter=",")

dummy_cls_person_4 = DummyClassifier()
k_neighbors_cls_person_4 = KNeighborsClassifier()
decision_tree_cls_person_4 = DecisionTreeClassifier()
random_forest_cls_person_4 = RandomForestClassifier()
mlp_cls_person_4 = MLPClassifier()
gaussian_nb_cls_person_4 = GaussianNB()

print('Test Classifiers on PERSON_4 learned with data collected by PERSON_4')
print('Dummy Classifier        ', sig.test_and_learn_classifier(dummy_cls_person_4, features_person_4))
print('K-Neighbors Classifier  ', sig.test_and_learn_classifier(k_neighbors_cls_person_4, features_person_4))
print('Decision Tree Classifier', sig.test_and_learn_classifier(decision_tree_cls_person_4, features_person_4))
print('Random Forest Classifier', sig.test_and_learn_classifier(random_forest_cls_person_4, features_person_4))
print('MLP Classifier          ', sig.test_and_learn_classifier(mlp_cls_person_4, features_person_4))
print('GaussianNB              ', sig.test_and_learn_classifier(gaussian_nb_cls_person_4, features_person_4))


features_all = np.loadtxt('mgr/data/resources/all/Features_all.csv', delimiter=",")

dummy_cls_all = DummyClassifier()
k_neighbors_cls_all = KNeighborsClassifier()
decision_tree_cls_all = DecisionTreeClassifier()
random_forest_cls_all = RandomForestClassifier()
mlp_cls_all = MLPClassifier()
gaussian_nb_cls_all = GaussianNB()

activity_features = features_all[:, 1:]
activity_markers = features_all[:, 0]
af_train, af_test, am_train, am_test = train_test_split(activity_features, activity_markers, test_size=.35)

dummy_cls_all.fit(af_train, am_train)
k_neighbors_cls_all.fit(af_train, am_train)
decision_tree_cls_all.fit(af_train, am_train)
random_forest_cls_all.fit(af_train, am_train)
mlp_cls_all.fit(af_train, am_train)
gaussian_nb_cls_all.fit(af_train, am_train)


print('Test Classifiers on PERSON_1 learned with all data')
print('Dummy Classifier        ', sig.test_classifier(dummy_cls_all, features_person_1))
print('K-Neighbors Classifier  ', sig.test_classifier(k_neighbors_cls_all, features_person_1))
print('Decision Tree Classifier', sig.test_classifier(decision_tree_cls_all, features_person_1))
print('Random Forest Classifier', sig.test_classifier(random_forest_cls_all, features_person_1))
print('MLP Classifier          ', sig.test_classifier(mlp_cls_all, features_person_1))
print('GaussianNB              ', sig.test_classifier(gaussian_nb_cls_all, features_person_1))

print('Test Classifiers on PERSON_2 learned with all data')
print('Dummy Classifier        ', sig.test_classifier(dummy_cls_all, features_person_2))
print('K-Neighbors Classifier  ', sig.test_classifier(k_neighbors_cls_all, features_person_2))
print('Decision Tree Classifier', sig.test_classifier(decision_tree_cls_all, features_person_2))
print('Random Forest Classifier', sig.test_classifier(random_forest_cls_all, features_person_2))
print('MLP Classifier          ', sig.test_classifier(mlp_cls_all, features_person_2))
print('GaussianNB              ', sig.test_classifier(gaussian_nb_cls_all, features_person_2))

print('Test Classifiers on PERSON_3 learned with all data')
print('Dummy Classifier        ', sig.test_classifier(dummy_cls_all, features_person_3))
print('K-Neighbors Classifier  ', sig.test_classifier(k_neighbors_cls_all, features_person_3))
print('Decision Tree Classifier', sig.test_classifier(decision_tree_cls_all, features_person_3))
print('Random Forest Classifier', sig.test_classifier(random_forest_cls_all, features_person_3))
print('MLP Classifier          ', sig.test_classifier(mlp_cls_all, features_person_3))
print('GaussianNB              ', sig.test_classifier(gaussian_nb_cls_all, features_person_3))


print('Test Classifiers on PERSON_4 learned with all data')
print('Dummy Classifier        ', sig.test_classifier(dummy_cls_all, features_person_4))
print('K-Neighbors Classifier  ', sig.test_classifier(k_neighbors_cls_all, features_person_4))
print('Decision Tree Classifier', sig.test_classifier(decision_tree_cls_all, features_person_4))
print('Random Forest Classifier', sig.test_classifier(random_forest_cls_all, features_person_4))
print('MLP Classifier          ', sig.test_classifier(mlp_cls_all, features_person_4))
print('GaussianNB              ', sig.test_classifier(gaussian_nb_cls_all, features_person_4))

### Sample

activity_features = np.loadtxt('mgr/data/resources/Features_activity.csv', delimiter=",")
print("Features end")
print(datetime.datetime.now())
#DRAW - WORKING
#data_to_predict = features[:, 1:] #dla danych treningowych
data_to_predict = activity_features
print("Prediction start")
print(datetime.datetime.now())
y = np.array(random_forest_cls_all.predict(data_to_predict))
x = np.linspace(0, len(data_to_predict)-1, len(data_to_predict))
print("Prediction end")
print(datetime.datetime.now())

data = np.column_stack((x,y))

STANDING_COLOR = 'gray'
WALKING_COLOR = 'green'
DOWNSTAIRS_COLOR = 'brown'
UPSTAIRS_COLOR = 'blue'
RUNNING_COLOR = 'red'

standing = data[data[:, 1] == 0]
walking = data[data[:, 1] == 1]
downstairs = data[data[:, 1] == 2]
upstairs = data[data[:, 1] == 3]
running = data[data[:, 1] == 4]
dot_size = 3
print("Printing")
print(datetime.datetime.now())
plt.scatter(standing[:, 0], standing[:,1], dot_size, c=STANDING_COLOR, label='standing')
plt.scatter(walking[:, 0], walking[:,1], dot_size, c=WALKING_COLOR, label='walking')
plt.scatter(downstairs[:, 0], downstairs[:,1], dot_size, c=DOWNSTAIRS_COLOR, label='downstairs')
plt.scatter(upstairs[:, 0], upstairs[:, 1], dot_size, c=UPSTAIRS_COLOR, label='upstairs')
plt.scatter(running[:, 0], running[:,1], dot_size, c=RUNNING_COLOR, label='running')
plt.legend(loc='best')
print("Show")
print(datetime.datetime.now())
plt.show()