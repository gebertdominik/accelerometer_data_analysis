import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.signal import lfilter, butter
from scipy.signal import lfilter_zi

import mgr.data.readrawdata as rd
import mgr.visualization.visualize as vis
import mgr.calc.signal as sig
import numpy as np
"""mu, sigma = 0, 500

x = np.arange(1, 100, 0.1)  # x axis
z = np.random.normal(mu, sigma, len(x))  # noise
y = x ** 2 + z # data
plt.plot(x, y, linewidth=2, linestyle="-", c="b")  # it include some noise
plt.show()


n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,y)
plt.plot(x, yy, linewidth=2, linestyle="-", c="b")  # smooth by filter
plt.show()
"""
# WALKING = rd.read('mgr/data/resources/20160109_lt.csv')
STANDING = rd.read('mgr/data/resources/ok/standing.csv')

STANDING['magnitude'] = sig.magnitude(STANDING)
#vis.graph_magnitude(STANDING, 'red')


def denoise(activity):
    n = 3
    b = [1/n] * n
    a = 1
    activity['magnituded'] = filtfilt(b,a,activity['yAxis'],padtype='constant')


denoise(STANDING)


def graph_magnitude_without_noise(activity):

    fig, (ax0,ax1) = plt.subplots(nrows=2, figsize=(15, 10), sharex=True)
    vis.plot_graph(ax0, activity['timestamp'], activity['yAxis'], 'with noise', 'red')
    vis.plot_graph(ax1, activity['timestamp'], activity['magnituded'], 'without noise filtfilt', 'red')
    plt.subplots_adjust(hspace=0.2)
    plt.show()

#vis.graph_activity(WALKING)
#print("printing")
#graph_magnitude_without_noise(WALKING)
graph_magnitude_without_noise(STANDING)




