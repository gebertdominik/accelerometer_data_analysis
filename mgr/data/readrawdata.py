import pandas as pd


def read(filename):
    columns = ['timestamp', 'xAxis', 'yAxis', 'zAxis']
    data = pd.read_csv(filename, skiprows=300, skipfooter=300, header=None, names=columns)
    return data


