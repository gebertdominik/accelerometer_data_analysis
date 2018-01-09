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
STANDING = rd.read('mgr/data/resources/ok/standing.csv')
"""vis.graph_activity(STANDING, STANDING_COLOR, 'Standing')"""

#Read walking activity data from csv and draw a graph
WALKING = rd.read('mgr/data/resources/ok/walking.csv')
"""vis.graph_activity(WALKING, WALKING_COLOR, 'Walking')"""

#Read downstairs activity data from csv and draw a graph
DOWNSTAIRS = rd.read('mgr/data/resources/ok/downstairs.csv')
"""vis.graph_activity(DOWNSTAIRS, DOWNSTAIRS_COLOR, 'Downstairs')"""

#Read upstairs activity data from csv and draw a graph
UPSTAIRS = rd.read('mgr/data/resources/ok/upstairs.csv')
"""vis.graph_activity(UPSTAIRS, UPSTAIRS_COLOR, 'Upstairs')"""

#Read running activity data from csv and draw a graph
RUNNING = rd.read('mgr/data/resources/ok/running.csv')
"""vis.graph_activity(RUNNING, RUNNING_COLOR, 'Running')"""

# Step 2 - remove noise from data and draw a graphs

#Denoise the activities
sig.denoise(STANDING)
sig.denoise(WALKING)
sig.denoise(DOWNSTAIRS)
sig.denoise(UPSTAIRS)
sig.denoise(RUNNING)

#Draw a graphs of denoised activities
"""
vis.graph_activity(STANDING, STANDING_COLOR, 'Standing - without noise')
vis.graph_activity(WALKING, WALKING_COLOR, 'Walking - without noise')
vis.graph_activity(DOWNSTAIRS, DOWNSTAIRS_COLOR, 'Downstairs - without noise')
vis.graph_activity(UPSTAIRS, UPSTAIRS_COLOR, 'Upstairs - without noise')
vis.graph_activity(RUNNING, RUNNING_COLOR, 'Running - without noise')
"""
# Step 3 - calculate magnitude and draw a graphs

#Calculate magnitudes
STANDING['magnitude'] = sig.magnitude(STANDING)
WALKING['magnitude'] = sig.magnitude(WALKING)
DOWNSTAIRS['magnitude'] = sig.magnitude(DOWNSTAIRS)
UPSTAIRS['magnitude'] = sig.magnitude(UPSTAIRS)
RUNNING['magnitude'] = sig.magnitude(RUNNING)
"""
#Draw a graphs of magnitudes
vis.graph_magnitude(STANDING, STANDING_COLOR, 'Standing - magnitude')
vis.graph_magnitude(WALKING, WALKING_COLOR, 'Walking - magnitude')
vis.graph_magnitude(DOWNSTAIRS, DOWNSTAIRS_COLOR, 'Downstairs - magnitude')
vis.graph_magnitude(UPSTAIRS, UPSTAIRS_COLOR, 'Upstairs - magnitude')
vis.graph_magnitude(RUNNING, RUNNING_COLOR, 'Running - magnitude')
"""

# Step 4 - draw divided signal


"""
vis.graph_divided_signal(STANDING, STANDING_COLOR, 'Standing - divided')
vis.graph_divided_signal(WALKING, WALKING_COLOR, 'Walking - divided')
vis.graph_divided_signal(DOWNSTAIRS, DOWNSTAIRS_COLOR, 'Downstairs - divided')
vis.graph_divided_signal(UPSTAIRS, UPSTAIRS_COLOR, 'Upstairs - divided')
vis.graph_divided_signal(RUNNING, RUNNING_COLOR, 'Running - divided')
"""

# Step 5 - extract features
"""
activities = [STANDING, WALKING, DOWNSTAIRS, UPSTAIRS, RUNNING]
output_file_path = 'mgr/data/resources/Features.csv'

with open(output_file_path, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities)):
        for f in sig.extract_features(activities[i]):
            rows.writerow([i] + f)

"""
# Step 6 - test classifiers

features = np.loadtxt('mgr/data/resources/Features.csv', delimiter=",")

dummy_cls = DummyClassifier()
k_neighbors_cls = KNeighborsClassifier()
decision_tree_cls = DecisionTreeClassifier()
random_forest_cls = RandomForestClassifier()
mlp_cls = MLPClassifier()
gaussian_nb_cls = GaussianNB()

print('Dummy Classifier        ', sig.test_and_learn_classifier(dummy_cls, features))
print('K-Neighbors Classifier  ', sig.test_and_learn_classifier(k_neighbors_cls, features))
print('Decision Tree Classifier', sig.test_and_learn_classifier(decision_tree_cls, features))
print('Random Forest Classifier', sig.test_and_learn_classifier(random_forest_cls, features))
print('MLP Classifier          ', sig.test_and_learn_classifier(mlp_cls, features))
print('GaussianNB              ', sig.test_and_learn_classifier(gaussian_nb_cls, features))




# Step 7 - all day activity

#print(datetime.datetime.now())
ACTIVITY = rd.read('mgr/data/20171219_allday_dg.csv')
#sig.denoise(ACTIVITY)
#ACTIVITY['magnitude'] = sig.magnitude(ACTIVITY)

#print("Features start")
#print(datetime.datetime.now())

#output_file_path = 'mgr/data/resources/Features_activity.csv'

#with open(output_file_path, 'w') as features_file:
#    rows = csv.writer(features_file)
#    for f in sig.extract_features(ACTIVITY):
 #       rows.writerow(f)

activity_features = np.loadtxt('mgr/data/resources/Features_activity.csv', delimiter=",")
print("Features end")
print(datetime.datetime.now())
#DRAW - WORKING
#data_to_predict = features[:, 1:] #dla danych treningowych
data_to_predict = activity_features
print("Prediction start")
print(datetime.datetime.now())
y = np.array(random_forest_cls.predict(data_to_predict))
x = np.linspace(0, len(data_to_predict)-1, len(data_to_predict))
print("Prediction end")
print(datetime.datetime.now())

data = np.column_stack((x,y))

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

