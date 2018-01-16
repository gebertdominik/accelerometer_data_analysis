import mgr.data.readrawdata as rd
import mgr.calc.signal as sig
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
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


activities_person_1 = [STANDING_PERSON_1, WALKING_PERSON_1, DOWNSTAIRS_PERSON_1, UPSTAIRS_PERSON_1, RUNNING_PERSON_1]
features_file_person_1 = 'mgr/data/resources/person_1/Features.csv'

with open(features_file_person_1, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_1)):
        for f in sig.extract_features(activities_person_1[i]):
            rows.writerow([i] + f)

""" PERSON_1 - prepare data - stop """
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



activities_person_2 = [STANDING_PERSON_2, WALKING_PERSON_2, DOWNSTAIRS_PERSON_2, UPSTAIRS_PERSON_2, RUNNING_PERSON_2]
features_file_person_2 = 'mgr/data/resources/person_2/Features.csv'

with open(features_file_person_2, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_2)):
        for f in sig.extract_features(activities_person_2[i]):
            rows.writerow([i] + f)

""" PERSON_2 - prepare data - stop """
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



activities_person_3 = [STANDING_PERSON_3, WALKING_PERSON_3, DOWNSTAIRS_PERSON_3, UPSTAIRS_PERSON_3, RUNNING_PERSON_3]
features_file_person_3 = 'mgr/data/resources/person_3/Features.csv'

with open(features_file_person_3, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_3)):
        for f in sig.extract_features(activities_person_3[i]):
            rows.writerow([i] + f)

""" PERSON_3 - prepare data - stop """
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
features_file_person_4 = 'mgr/data/resources/person_4/Features.csv'

with open(features_file_person_4, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_4)):
        for f in sig.extract_features(activities_person_4[i]):
            rows.writerow([i] + f)

