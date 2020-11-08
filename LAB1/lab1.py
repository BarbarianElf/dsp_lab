"""
lab 1 for DSP LAB

Created on October 23rd 2020

@authors: Niv Ben Ami & Ziv Zango
"""
import csv
import numpy
import librosa
import scipy
import soundfile
import os

import lab_utils as lu


def q3_pre(r):
    try:
        with open("results/fft_time.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["dftmtx time",	"fft time", "fastest", "trans equals"])
            for index in r:
                x = list(range(1, index + 1))
                y_dftmtx, t1 = lu.dftmtx_mul_x(x)
                y_fft, t2 = lu.fft(x)
                fastest = "dftmtx" if t1 < t2 else "fft"
                writer.writerow([t1, t2, fastest, "Yes" if y_dftmtx.all() == y_fft.all() else "No"])
        return True
    except Exception as err:
        print(f"cannot proceed with err: {err}")
        return False


def q5_pre(vector_len):
    rand_v = lu.random_vector(vector_len)
    y1 = lu.regular_fft_ifft(rand_v)
    y2 = lu.conj_fft_ifft(rand_v)
    if (y1 == y2).all():
        return True
    return False


def q1(plt, vector_len=8):
    t = numpy.arange(0, vector_len, 1)
    rand_v = lu.random_vector(vector_len)
    y1 = lu.conj_dft_idft_with_dftmtx(rand_v)
    fig, (sub1, sub2) = plt.subplots(2)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Comparison between 2 algorithms")
    sub1.plot(t, numpy.abs(rand_v))
    sub1.set_title("original signal")
    sub2.plot(t, numpy.abs(y1))
    sub2.set_title("after conj-dftmtx-conj")
    plt.savefig("results/comp_2_algo")
    return True


def q2(fs, number_of_points, table1_harmonics):
    t = numpy.arange(0, number_of_points/fs, 1/fs)
    f = numpy.arange(-fs/2, fs/2, fs/number_of_points)
    x = 17*numpy.sin(t*2*numpy.pi*120) + 5.1*numpy.sin(t*2*numpy.pi*600)
    y, t1 = lu.dftmtx_mul_x(x)
    y = numpy.abs(numpy.fft.fftshift(y / number_of_points))

    # writing table 1
    try:
        with open("results/dft_q2.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Harmonic[Hz]", "DFT result"])
            for i in range(len(table1_harmonics)):
                index = lu.find_freq_index(table1_harmonics[i], f)
                writer.writerow([f[index], y[index]])
    except Exception as err:
        print(f"cannot proceed with err: {err}")
        return False

    fig1 = lu.get_plot(t, x, title=f"x(t) in time domain\nN={number_of_points}", title_x="t[s]", title_y="x(t)",
                       save_name=f"results/q2-t-{number_of_points}")
    fig2 = lu.get_stem(f, y, title=f"X(f) DFT\nN={number_of_points}", title_x="f[Hz]", title_y="X(f)",
                       limx=[-800, 800], save_name=f"results/q2-dft-{number_of_points}")
    return True


def q3(fs, number_of_points):
    nt = numpy.arange(0, number_of_points / fs, 1/fs)
    f = numpy.arange(-fs / 2, fs / 2, fs / number_of_points)
    x = 2 + numpy.sin(nt*2*numpy.pi*20) + numpy.sin(nt*2*numpy.pi*41)
    y_dit = lu.dit_fft_radix2(x) / len(x)
    y_fft = numpy.fft.fft(x) / len(x)
    y_dif = lu.dif_fft_radix2(x) / len(x)
    if (numpy.round(numpy.abs(y_dif), 10) == numpy.round(numpy.abs(y_fft), 10)).all() and \
            (numpy.round(numpy.abs(y_dit), 10) == numpy.round(numpy.abs(y_fft), 10)).all():
        print("SAME FFT")
    y_dit = numpy.fft.fftshift(numpy.abs(y_dit))
    print(f"DIT sums: {lu.dit_sum_count}\nDIT mults: {lu.dit_mul_count}")
    print(f"DIF sums: {lu.dif_sum_count}\nDIF mults: {lu.dif_mul_count}")

    fig1 = lu.get_stem(f, y_dit, title=f"X(f) fft radix2 DIT\nN={number_of_points}", title_x="f[Hz]", title_y="X(f)",
                       save_name=f"results/q3-fftradix2-{number_of_points}")
    return


def q5(wav_data, fs, number_of_points, noise_frequency):
    nt = numpy.arange(0, number_of_points / fs, 1/fs)
    f = numpy.arange(-fs / 2, fs / 2, fs / number_of_points)
    noise = (1/100)*numpy.sin(nt*2*numpy.pi*noise_frequency)
    signal_input_t = noise + wav_data
    noise = numpy.fft.fftshift(numpy.fft.fft(noise)) / number_of_points
    wav_f = numpy.fft.fftshift(numpy.fft.fft(wav_data)) / number_of_points
    signal_input_f = noise + wav_f

    fir_coeff = scipy.signal.firwin(3999, [noise_frequency-20, noise_frequency+20], pass_zero='bandstop', fs=fs)
    filtered_signal = scipy.signal.lfilter(fir_coeff, 1.0, signal_input_t)
    filtered_signal_f = numpy.fft.fftshift(numpy.fft.fft(filtered_signal)) / number_of_points

    o = lu.filter_using_utils(signal_input_t, fir_coeff)
    o1 = numpy.real(o[0:number_of_points])

    soundfile.write("results/with_noise.wav", signal_input_t, fs)
    soundfile.write("results/after_filter1.wav", filtered_signal, fs)
    soundfile.write("results/after_filter2.wav", o1, fs)

    fig0 = lu.get_plot(nt, wav_data, title="sound time domain", title_x="t", title_y="amp",
                       save_name="results/q5-time-before-noise")
    fig1 = lu.get_stem(f, numpy.abs(wav_f), title="input spec without noise", title_x="f[Hz]", title_y="amp",
                       limx=[-4000, 4000], save_name="results/q5-freq-before-noise")
    fig2 = lu.get_stem(f, numpy.abs(signal_input_f), title="input spec with noise", title_x="f[Hz]", title_y="amp",
                       limx=[-4000, 4000], save_name="results/q5-freq-with-noise")
    fig4 = lu.get_stem(f, numpy.abs(filtered_signal_f), title="filtered signal", title_x="f[Hz]", title_y="amp",
                       limx=[-4000, 4000], save_name="results/q5-freq-filtered-signal")
    fig5 = lu.get_plot(nt, filtered_signal, title="sound after filter 1", title_x="t", title_y="amp",
                       save_name="results/q-time-after-filter1")
    fig6 = lu.get_plot(nt, o1, title="sound after filter 2", title_x="t", title_y="amp",
                       save_name="results/q-time-after-filter2")
    return


def q_vad(wav_data, fs, number_of_points):
    nt = numpy.arange(0, number_of_points / fs, 1 / fs)
    vad = numpy.zeros_like(sound_data)
    for init, end in librosa.effects.split(sound_data, top_db=30):
        vad[init:end].fill(1)
    vad_utils = lu.vad_led(wav_data, fs, number_of_points, frame_size=0.02, k=2, p=0.1)
    fig0 = lu.get_plot(nt, wav_data, vad, title="VAD plot", title_x="t[s]", title_y="amp",
                       save_name="results/VAD-plot")
    fig1 = lu.get_plot(nt, wav_data, vad_utils, title="VAD UTILS plot", title_x="t[s]", title_y="amp",
                       save_name="results/VAD-UTILS-plot")
    return


if __name__ == "__main__":
    plt = lu.get_plt()
    os.makedirs("results", exist_ok=True)

    # preparation to LAB1
    q3_pre(r=[pow(2, i + 2) for i in range(1, 10)])
    if q5_pre(vector_len=16):
        print("EQUAL for Q5-PRE")

    # LAB1
    if q1(vector_len=8, plt=plt):
        print("EQUAL for Q1")
    q2(fs=1e5, number_of_points=2500, table1_harmonics=[120 * i for i in range(1, 8)])
    q3(fs=128, number_of_points=64)
    sound_data, frequency_sampled = librosa.core.load("dsp_record-1.wav", sr=None)
    q5(sound_data, frequency_sampled, len(sound_data), noise_frequency=1000)
    q_vad(sound_data, frequency_sampled, len(sound_data))

    # Unmark the command blow to show ALL the figures
    # plt.show()
