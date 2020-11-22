"""
Exrta (lab 3) for DSP LAB

Created on November 22th 2020

@authors: Niv Ben Ami & Ziv Zango
"""
import numpy
import adaptfilt
import matplotlib.pyplot as plt
import pandas as pd

from lmsfilter import *

csv_file_name = "datasets_1840_3212_Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv"


def parse_file_lab3(number_of_samples):
    df = pd.read_csv(csv_file_name,
                     parse_dates=['Date'],
                     index_col='Date',
                     date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'))
    df.sort_index(inplace=True)
    ts = df[pd.Series(pd.to_datetime(df.index, errors='coerce')).notnull().values]
    ts['Close'] = pd.to_numeric(ts['Close'], errors='coerce')
    ts.dropna(inplace=True)
    ts.drop(['Open', 'High', 'Volume', 'Low', 'Stock Trading'], axis=1, inplace=True)
    out_array = ts['Close'].values.flatten()[: number_of_samples]
    return out_array


def initialize_inputs(samples, delay):
    signal = parse_file_lab3(samples)
    desired = signal[delay:]
    return signal[:-5], desired


def comparison(u, d, filter_length, step_size, type):
    if type is "NLMS":
        y0, e0, w0 = LMSFilter(u, d, filter_length, step_size).normalized()
        y1, e1, w1 = adaptfilt.nlms(u, d, filter_length, step_size)
    elif type is "LMS":
        y0, e0, w0 = LMSFilter(u, d, filter_length, step_size).regular()
        y1, e1, w1 = adaptfilt.lms(u, d, filter_length, step_size)
    else:
        print("you must specify type of comparison: NLMS, LMS")
        return False
    fig1, axs = plt.subplots(3, 2, sharex='all')
    plt.subplots_adjust(hspace=0.35)
    fig1.suptitle(f"Comparison between 2 algorithms {type} filter")
    axs[0, 0].plot(d, 'tab:blue')
    axs[0, 0].set_title('desired signal')
    axs[1, 0].plot(y0, 'tab:purple')
    axs[1, 0].set_title('my output')
    axs[2, 0].plot(e0, 'tab:red')
    axs[2, 0].set_title('my error')
    axs[0, 1].plot(d, 'tab:blue')
    axs[0, 1].set_title('desired signal')
    axs[1, 1].plot(y1, 'tab:purple')
    axs[1, 1].set_title('adaptfilt output')
    axs[2, 1].plot(e1, 'tab:red')
    axs[2, 1].set_title('adaptfilt error')
    for ax in axs.flat:
        ax.label_outer()
        ax.grid(True)
    plt.savefig(f"{type}-comparison")
    return True


if __name__ == "__main__":
    number_of_points = 1000
    delay = 5
    u, d = initialize_inputs(number_of_points, delay=delay)

    # Test LMSFilter normalized method comparison with adaptfilt nlms
    comparison(u, d, 32, 1, type="NLMS")

    # Test LMSFilter regular method comparison with adaptfilt lms (sinwave fs=8000)
    fs = 8000
    nt = numpy.arange(0, number_of_points / fs, 1 / fs)
    s = numpy.sin(nt * 2 * numpy.pi * 200)
    d = s + numpy.random.rand(number_of_points) * numpy.sqrt(0.1)
    u = numpy.pad(d[0:-delay], (delay, 0))
    comparison(u, d, 32, 0.01, type="LMS")

    # Unmark the command blow to show ALL the figures
    plt.show()
