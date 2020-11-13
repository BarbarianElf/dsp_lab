"""
lab 3 for DSP LAB

Created on November 13th 2020

@authors: Niv Ben Ami & Ziv Zango
"""
import numpy
import adaptfilt

import lab_utils as lu


def plot_adaptive_filter_lms(u, d, filter_length, mu):
    global plt
    y, e, w = adaptfilt.lms(u, d, filter_length, mu)
    fig, axs = plt.subplots(3, sharex=True)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(f"LMS algorithm step size: {mu}  filter length: {filter_length}")
    axs[0].plot(d)
    axs[0].set_title('Sine-wave corrupted by noise')
    axs[1].plot(y)
    axs[1].set_title('Cleaned sine-wave')
    axs[2].plot(e)
    axs[2].set_title('error signal')
    fig_psd, axp = plt.subplots(2, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.5)
    fig_psd.suptitle(f"PSD step size: {mu}  filter length: {filter_length}")
    axp[0].psd(d, Fs=8000)
    axp[0].grid(True)
    axp[0].set_title('PSD before')
    axp[1].psd(y, Fs=8000)
    axp[1].grid(True)
    axp[1].set_title('PSD after')
    return


def q3_and_q4(fs, number_of_points, step_size_array, filter_coeffs_length_array, delay=1):
    nt = numpy.arange(0, number_of_points / fs, 1 / fs)
    s = numpy.sin(nt * 2 * numpy.pi * 200)
    d = s + lu.random_vector(number_of_points) * numpy.sqrt(0.1)
    x = numpy.pad(d[0:-1], (delay, 0))
    for step_size in step_size_array:
        plot_adaptive_filter_lms(x, d, 16, step_size)
    for filter_coeffs_length in filter_coeffs_length_array:
        plot_adaptive_filter_lms(x, d, filter_coeffs_length, 0.01)
    return


def q5(fs, number_of_points, harmonic_factor_array, step_size=0.01, filter_coeffs_length=16, delay=1):
    nt = numpy.arange(0, number_of_points / fs, 1 / fs)
    for harmonic_factor in harmonic_factor_array:
        s = numpy.sin(nt * 2 * numpy.pi * 200) + numpy.sin(nt * 2 * numpy.pi * 200 * harmonic_factor)
        d = s + lu.random_vector(number_of_points) * numpy.sqrt(0.1)
        x = numpy.pad(d[0:-1], (delay, 0))
        plot_adaptive_filter_lms(x, d, filter_coeffs_length, step_size)
    return


if __name__ == "__main__":
    plt = lu.get_plt()
    plt.rcParams.update({'figure.max_open_warning': 0})
    q3_and_q4(fs=8000,
              number_of_points=500,
              step_size_array=[0.01, 0.2, 0.1, 0.005, 0.001],
              filter_coeffs_length_array=[32, 64, 128])
    q5(fs=8000,
       number_of_points=500,
       harmonic_factor_array=[2, 3, 4])
    plt.show()
