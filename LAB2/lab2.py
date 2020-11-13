"""
lab 2 for DSP LAB

Created on November 4th 2020

@authors: Niv Ben Ami & Ziv Zango
"""
import os
import numpy
import scipy.signal

import lab_utils as lu


def q1_or_q2_pre(x, h):
    demanded_result = numpy.correlate(x, h, "full")
    result, t = lu.cross_correlation(x, h)
    if (result == demanded_result).all():
        return True
    return False


def q3_pre(x, h):
    y = numpy.convolve(x, h)
    r = numpy.correlate(x, h, "full")
    print(f"Cross-Correlation result:\n{r}\nConvolution result:\n{y}")
    return True


def q4_pre(rect1, rect2, rect3, plt):
    fig1, axs = plt.subplots(3, 2, sharex=True)
    plt.subplots_adjust(hspace=0.35)
    fig1.suptitle("Comparison between 2 algorithms")
    axs[0, 0].plot(numpy.correlate(rect1, rect1, mode="full"), 'tab:blue')
    axs[0, 0].set_title('Auto correlation rect1')
    axs[1, 0].plot(numpy.correlate(rect2, rect2, mode="full"), 'tab:olive')
    axs[1, 0].set_title('Auto correlation rect2')
    axs[2, 0].plot(numpy.correlate(rect3, rect3, mode="full"), 'tab:red')
    axs[2, 0].set_title('Auto correlation rect3')
    axs[0, 1].plot(numpy.correlate(rect1, rect2, mode="full"), 'tab:green')
    axs[0, 1].set_title('Cross correlation rect1 rect2')
    axs[1, 1].plot(numpy.correlate(rect1, rect3, mode="full"), 'tab:purple')
    axs[1, 1].set_title('Cross correlation rect1 rect3')
    axs[2, 1].plot(numpy.correlate(rect2, rect3, mode="full"), 'tab:orange')
    axs[2, 1].set_title('Cross correlation rect2 rect3')
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig("results/q4_pre")
    return True


def q1_and_q2(fs, number_of_points, plt):
    t = numpy.arange(0, number_of_points / fs, 1 / fs)
    f = numpy.arange(-fs / 2, fs / 2, fs / number_of_points)
    x = numpy.sin(t * 2 * numpy.pi * 1000)
    awgn = numpy.random.normal(0, 1, number_of_points)
    auto_corr = numpy.correlate(x, x, mode='full')
    xcorr = numpy.correlate(x, awgn, mode='full')
    fig1, axs = plt.subplots(2, 2, sharex=True)
    plt.subplots_adjust(hspace=0.35)
    axs[0, 0].plot(t, x, 'tab:blue')
    axs[0, 0].set_title('Sin')
    axs[1, 0].plot(numpy.arange(0, len(auto_corr) / fs, 1 / fs), auto_corr, 'tab:blue')
    axs[1, 0].set_title('Sin Auto-correlation')
    axs[0, 1].plot(t, awgn, 'tab:red')
    axs[0, 1].set_title('AWGN')
    axs[1, 1].plot(numpy.arange(0, len(xcorr) / fs, 1 / fs), xcorr, 'tab:purple')
    axs[1, 1].set_title('Cross-correlation SIN-AWGN')
    plt.savefig("results/q1_and_q2")
    return


def q3(fs, number_of_points, plt):
    t = numpy.arange(0, number_of_points / fs, 1 / fs)
    f = numpy.arange(-fs / 2, fs / 2, fs / number_of_points)
    x = scipy.signal.square(t * 2 * numpy.pi * 1000)
    awgn = numpy.random.normal(0, 1, number_of_points)
    xcorr = numpy.correlate(x, awgn, mode='full')
    conv = numpy.convolve(x, awgn)
    fig1, axs = plt.subplots(2, 2, sharex=True)
    plt.subplots_adjust(hspace=0.35, wspace=0.35)
    axs[0, 0].plot(t, x, 'tab:blue')
    axs[0, 0].set_title('square')
    axs[1, 0].plot(numpy.arange(0, len(xcorr) / fs, 1 / fs), xcorr, 'tab:purple')
    axs[1, 0].set_title('Cross-correlation SQAURE-AWGN')
    axs[0, 1].plot(t, awgn, 'tab:red')
    axs[0, 1].set_title('AWGN')
    axs[1, 1].plot(numpy.arange(0, len(conv) / fs, 1 / fs), conv, 'tab:purple')
    axs[1, 1].set_title('Convolution SQAURE-AWGN')
    plt.savefig("results/q3")
    return


def q4(s1, s2, s3, s4, plt):
    r12 = numpy.correlate(s1, s2, mode="full")
    r34 = numpy.correlate(s3, s4, mode="full")

    p12 = r12/numpy.sqrt(lu.short_term_energy(s1) * lu.short_term_energy(s2))
    p34 = r34/numpy.sqrt(lu.short_term_energy(s3) * lu.short_term_energy(s4))

    fig1, axs = plt.subplots(2, 2, sharex=True)
    plt.subplots_adjust(hspace=0.35)
    axs[0, 0].plot(r12, 'tab:blue')
    axs[0, 0].set_title('r12')
    axs[0, 1].plot(r34, 'tab:red')
    axs[0, 1].set_title('r34')
    axs[1, 0].plot(p12, 'tab:blue')
    axs[1, 0].set_title('p12')
    axs[1, 1].plot(p34, 'tab:red')
    axs[1, 1].set_title('p34')
    plt.savefig("results/q4")
    return


def q5(s1, s2):
    res_fft, t1 = lu.cross_correlation_fft(s1, s2)
    res_fft = numpy.round(numpy.real(res_fft), 3)
    res_corr, t2 = lu.cross_correlation(s1, s2)
    print(f"FFT result: {res_fft}\ntime: {t1}\nresult: {res_corr}\ntime: {t2}")
    return


if __name__ == "__main__":
    plt = lu.get_plt()
    os.makedirs("results", exist_ok=True)

    # preparation to LAB2
    if q1_or_q2_pre(x=[1, 2, 3, 4], h=[4, 3, 2, 1]):
        print("EQUAL for Q1-PRE")
    if q1_or_q2_pre([1, 2, 3, 4], [1, 2, 3, 4]):
        print("EQUAL for Q2-PRE")
    q3_pre(x=[1, 2, 4], h=[1, 1, 1, 1])
    q3_pre(x=[0, 1, -2, 3, -4], h=[0.5, 1, 2, 1, 1, 0.5])
    q4_pre(1.5*scipy.signal.boxcar(5), 2*scipy.signal.boxcar(10), scipy.signal.boxcar(15), plt)

    # LAB2
    q1_and_q2(fs=2e5, number_of_points=500, plt=plt)
    q3(fs=1e5, number_of_points=450, plt=plt)
    q4(s1=[0, 3, 5, 5, 5, 2, 0.5, 0.25, 0],
       s2=[1, 1, 1, 1, 1, 0, 0, 0, 0],
       s3=[0, 9, 15, 15, 15, 6, 1.5, 0.75, 0],
       s4=[2, 2, 2, 2, 2, 0, 0, 0, 0], plt=plt)
    q5(s1=[8, 3, 5, 0.1],
       s2=[1, 6, 0.7, 1])

    # Unmark the command blow to show ALL the figures
    # plt.show()
