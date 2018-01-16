from prepare_data import *

activities_walking = [WALKING_PERSON_1, WALKING_PERSON_2, WALKING_PERSON_3, WALKING_PERSON_4]

output_file_path_walking = 'walking/Features_walking.csv'

with open(output_file_path_walking, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_walking)):
        for f in sig.extract_features(activities_walking[i]):
            rows.writerow([i] + f)

features_walking = np.loadtxt('walking/Features_walking.csv', delimiter=",")

print('Dummy Classifier        ', sig.test_dummy_cls_one_set(features_walking))
print('K-Neighbors Classifier  ', sig.test_knn_cls_one_set(features_walking))
print('Decision Tree Classifier', sig.test_decision_tree_cls_one_set(features_walking))
print('Random Forest Classifier', sig.test_random_forest_cls_one_set(features_walking))
print('MLP Classifier          ', sig.test_mlp_cls_one_set(features_walking))
print('GaussianNB              ', sig.test_gaussian_nb_cls_one_set(features_walking))