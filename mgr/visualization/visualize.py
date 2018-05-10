import matplotlib.pyplot as plt
import mgr.calc.signal as sig
import datetime
import matplotlib.dates as mdates


def graph_activity(activity, color, title):
    """Prints a graph of passed activity. There are 3 plots for each axis"""
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_graph(ax0, activity['timestamp'].values, activity['xAxis'], 'Oś X', color)
    plot_graph(ax1, activity['timestamp'].values, activity['yAxis'], 'Oś Y', color)
    plot_graph(ax2, activity['timestamp'].values, activity['zAxis'], 'Oś Z', color)
    #plt.suptitle(title)
    #plt.ylabel(" przysp.[$m/s^2$]")
    plt.tight_layout()
    plt.show()


def plot_graph(axis, xAxis, yAxis, title, color):
    """Function for single graph ploting"""
    axis.set_xlabel("czas [$s$]")
    axis.set_ylabel(" przysp.[$m/s^2$]")

    xAxis = xAxis - min(xAxis);
    axis.plot(xAxis, yAxis, color=color)
    axis.set_title(title, loc='left')
    #axis.xaxis.set_visible(False)
    axis.set_xlim(0,max(xAxis) - [min(xAxis)])
    axis.grid(True)


def graph_magnitude(activity, color):
    """Prints a activity magnitude graph"""
    fig, axs = plt.subplots(nrows=1, figsize=(15, 10), sharex=True)
    plot_graph(axs, activity['timestamp'], activity['magnitude'], "Wypadkowy wektor przyspieszenia", color)
    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    plt.show()


def graph_divided_signal(activity, color):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(15, 10), sharex=True)
    plot_graph(ax0, activity['timestamp'], activity['xAxis'], 'Oś X', color)
    plot_graph(ax1, activity['timestamp'], activity['yAxis'], 'Oś Y', color)
    plot_graph(ax2, activity['timestamp'], activity['zAxis'], 'Oś Z', color)
    plot_graph(ax3, activity['timestamp'], activity['magnitude'], 'Wypadkowy wektor przyspieszenia', color)

    for (start, end) in sig.divide_signal(activity['timestamp']):
        ax0.axvline(activity['timestamp'][start], color='black')
        ax1.axvline(activity['timestamp'][start], color='black')
        ax2.axvline(activity['timestamp'][start], color='black')
        ax3.axvline(activity['timestamp'][start], color='black')

    plt.tight_layout()
    plt.show()
    #plt.savefig('myfig.png')