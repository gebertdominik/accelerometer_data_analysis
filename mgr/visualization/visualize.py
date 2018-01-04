
import matplotlib.pyplot as plt
import mgr.calc.signal as sig


def graph_activity(activity, color, title):
    """Prints a graph of passed activity. There are 3 plots for each axis"""
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_graph(ax0, activity['timestamp'], activity['xAxis'], 'X-axis', color)
    plot_graph(ax1, activity['timestamp'], activity['yAxis'], 'Y-axis', color)
    plot_graph(ax2, activity['timestamp'], activity['zAxis'], 'Z-axis', color)
    plt.suptitle(title)
    plt.show()


def plot_graph(axis, xAxis, yAxis, title, color):
    """Function for single graph ploting"""
    axis.plot(xAxis, yAxis, color=color)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_xlim([min(xAxis), max(xAxis)])
    axis.grid(True)


def graph_magnitude(activity, color, title):
    """Prints a activity magnitude graph"""
    fig, axs = plt.subplots(nrows=1, figsize=(15, 10), sharex=True)
    plot_graph(axs, activity['timestamp'], activity['magnitude'], title, color)
    plt.subplots_adjust(hspace=0.2)
    plt.show()


def graph_divided_signal(activity, color, title):
    fig,(ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(15, 10), sharex=True)
    plot_graph(ax0, activity['timestamp'], activity['xAxis'], 'X-axis(divided)', color)
    plot_graph(ax1, activity['timestamp'], activity['yAxis'], 'Y-axis(divided)', color)
    plot_graph(ax2, activity['timestamp'], activity['zAxis'], 'Z-axis(divided)', color)
    plot_graph(ax3, activity['timestamp'], activity['magnitude'], 'magnitude(divided)', color)

    for (start, end) in sig.divide_signal(activity['timestamp']):
        ax0.axvline(activity['timestamp'][start], color='black')
        ax1.axvline(activity['timestamp'][start], color='black')
        ax2.axvline(activity['timestamp'][start], color='black')
        ax3.axvline(activity['timestamp'][start], color='black')

    plt.suptitle(title)
    plt.show()
