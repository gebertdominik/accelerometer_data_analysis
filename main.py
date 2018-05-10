from scipy.signal import lfilter
import mgr.calc.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import mgr.data.readrawdata as rd
import mgr.visualization.visualize as vis

# Step 1 - read and draw raw data

STANDING_COLOR = 'gray'
WALKING_COLOR = 'green'
DOWNSTAIRS_COLOR = 'brown'
UPSTAIRS_COLOR = 'blue'
RUNNING_COLOR = 'red'

#Read standing activity data from csv and draw a graph
STANDING = rd.read('mgr/data/resources/person_1/standing.csv')
vis.graph_activity(STANDING, STANDING_COLOR, 'Standing')

#Read walking activity data from csv and draw a graph
WALKING = rd.read('mgr/data/resources/person_1/walking.csv')
vis.graph_activity(WALKING, WALKING_COLOR, 'Walking')

#Read downstairs activity data from csv and draw a graph
DOWNSTAIRS = rd.read('mgr/data/resources/person_1/downstairs.csv')
vis.graph_activity(DOWNSTAIRS, DOWNSTAIRS_COLOR, 'Downstairs')

#Read upstairs activity data from csv and draw a graph
UPSTAIRS = rd.read('mgr/data/resources/person_1/upstairs.csv')
vis.graph_activity(UPSTAIRS, UPSTAIRS_COLOR, 'Upstairs')

#Read running activity data from csv and draw a graph
RUNNING = rd.read('mgr/data/resources/person_1/running.csv')
vis.graph_activity(RUNNING, RUNNING_COLOR, 'Running')



#Draw a graphs of denoised activities

#vis.graph_activity(STANDING, STANDING_COLOR, 'Standing - without noise')
#vis.graph_activity(WALKING, WALKING_COLOR, 'Walking - without noise')
#vis.graph_activity(DOWNSTAIRS, DOWNSTAIRS_COLOR, 'Downstairs - without noise')
#vis.graph_activity(UPSTAIRS, UPSTAIRS_COLOR, 'Upstairs - without noise')
#vis.graph_activity(RUNNING, RUNNING_COLOR, 'Running - without noise')

# Step 3 - calculate magnitude and draw a graphs

#Calculate magnitudes
STANDING['magnitude'] = sig.magnitude(STANDING)
WALKING['magnitude'] = sig.magnitude(WALKING)
DOWNSTAIRS['magnitude'] = sig.magnitude(DOWNSTAIRS)
UPSTAIRS['magnitude'] = sig.magnitude(UPSTAIRS)
RUNNING['magnitude'] = sig.magnitude(RUNNING)


#Draw a graphs of magnitudes
vis.graph_magnitude(STANDING, STANDING_COLOR)
vis.graph_magnitude(WALKING, WALKING_COLOR)
vis.graph_magnitude(DOWNSTAIRS, DOWNSTAIRS_COLOR)
vis.graph_magnitude(UPSTAIRS, UPSTAIRS_COLOR)
vis.graph_magnitude(RUNNING, RUNNING_COLOR)

# Step 4 - draw divided signal

vis.graph_divided_signal(STANDING, STANDING_COLOR)
vis.graph_divided_signal(WALKING, WALKING_COLOR)
vis.graph_divided_signal(DOWNSTAIRS, DOWNSTAIRS_COLOR)
vis.graph_divided_signal(UPSTAIRS, UPSTAIRS_COLOR)
vis.graph_divided_signal(RUNNING, RUNNING_COLOR)
"""

# Step 5 - extract features
"""
activities = [STANDING, WALKING, DOWNSTAIRS, UPSTAIRS, RUNNING]
output_file_path = 'mgr/data/features/person_1/Features.csv'

with open(output_file_path, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities)):
        for f in sig.extract_features(activities[i]):
            rows.writerow([i] + f)


# Step 6 - test classifier

features = np.loadtxt('mgr/data/features/person_1/Features.csv', delimiter=",")
k_neighbors_cls = KNeighborsClassifier()
decision_tree_cls = DecisionTreeClassifier()
random_forest_cls = RandomForestClassifier()
mlp_cls = MLPClassifier()
gaussian_nb_cls = GaussianNB()

print('K-Neighbors Classifier  ', sig.test_and_learn_classifier(k_neighbors_cls, features))
print('Decision Tree Classifier', sig.test_and_learn_classifier(decision_tree_cls, features))
print('Random Forest Classifier', sig.test_and_learn_classifier(random_forest_cls, features))
print('MLP Classifier          ', sig.test_and_learn_classifier(mlp_cls, features))
print('GaussianNB              ', sig.test_and_learn_classifier(gaussian_nb_cls, features))


