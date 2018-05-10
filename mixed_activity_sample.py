from prepare_data import *

print(datetime.datetime.now())
ACTIVITY = rd.read('mgr/data/resources/person_3/mixed_activity.csv')
ACTIVITY['magnitude'] = sig.magnitude(ACTIVITY)

print("Features start")
print(datetime.datetime.now())

output_file_path = 'mgr/data/features/person_3/Features_mixed_activity.csv'

with open(output_file_path, 'w') as features_file:
    rows = csv.writer(features_file)
    for f in sig.extract_features(ACTIVITY):
        rows.writerow(f)

activity_features = np.loadtxt('mgr/data/features/person_3/Features_mixed_activity.csv', delimiter=",")
print("Features end")
print(datetime.datetime.now())

#Learn cls
features_person_3 = np.loadtxt(features_file_person_3, delimiter=",")
activity_features_learn = features_person_3[:, 1:]
activity_markers_learn = features_person_3[:, 0]
cls = RandomForestClassifier()
cls.fit(activity_features_learn, activity_markers_learn)

#DRAW - WORKING
#data_to_predict = features[:, 1:] #dla danych treningowych
data_to_predict = activity_features
print("Prediction start")
print(datetime.datetime.now())
y = np.array(cls.predict(data_to_predict))
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
plt.scatter(standing[:, 0], standing[:,1], dot_size, c=STANDING_COLOR, label='stanie')
plt.scatter(walking[:, 0], walking[:,1], dot_size, c=WALKING_COLOR, label='chodzenie')
plt.scatter(downstairs[:, 0], downstairs[:,1], dot_size, c=DOWNSTAIRS_COLOR, label='schodzenie ze schodow')
plt.scatter(upstairs[:, 0], upstairs[:, 1], dot_size, c=UPSTAIRS_COLOR, label='wchodzenie po schodach')
plt.scatter(running[:, 0], running[:,1], dot_size, c=RUNNING_COLOR, label='bieganie')
plt.legend(loc='best')
print("Show")
print(datetime.datetime.now())
plt.tight_layout()
plt.show()

