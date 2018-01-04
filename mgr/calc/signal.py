import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cross_validation import train_test_split
import math
from scipy.signal import filtfilt


def denoise(activity):
    n = 3
    b = [1 / n] * n
    a = 1
    activity['xAxis'] = filtfilt(b, a, activity['xAxis'])
    activity['yAxis'] = filtfilt(b, a, activity['yAxis'])
    activity['zAxis'] = filtfilt(b, a, activity['zAxis'])


def magnitude(activity):
    res = np.sqrt(activity['xAxis'] * activity['xAxis'] +
                  activity['yAxis'] * activity['yAxis'] +
                  activity['zAxis'] * activity['zAxis'])
    return res


def divide_signal(df, size=100):
    start = 0
    while start < df.count():
        yield start, start + size
        start += (size / 2)


def window_values(axis, start, end):
    start = int(start)
    end = int(end)
    p25 = np.percentile(axis, 25)
    median = np.percentile(axis, 50)
    p75 = np.percentile(axis, 75)
    return [
        axis[start:end].mean(),
        p25,
        median,
        p75,
        axis[start:end].std(),
        axis[start:end].var(),
        axis[start:end].min(),
        axis[start:end].max(),
        skew(axis[start:end]),
        kurtosis(axis[start:end]),
    ]


def extract_features(activity):
    for (start, end) in divide_signal(activity['timestamp']):
        activity_values = ['xAxis', 'yAxis', 'zAxis', 'magnitude']
        features = []
        for axis in activity_values:
            features += window_values(activity[axis], start, end)
        yield features


def test_classifier(classifier, features_data):
    activity_features = features_data[:, 1:]
    activity_markers = features_data[:, 0]
    results =[]
    loops = 10
    for i in range(0, loops):
        af_train, af_test, am_train, am_test = train_test_split(activity_features, activity_markers, test_size=.35)
        classifier.fit(af_train, am_train)
        res = classifier.score(af_test, am_test)
        results.append(res)
    return[
        np.mean(results),
        np.std(results)
    ]

