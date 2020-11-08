'''
utils for DSP LAB

date created: October 23rd 2020

authors: Niv Ben Ami & Ziv Zango
'''
import matplotlib.pyplot as plt
import numpy
import functools
import time

dit_sum_count, dit_mul_count = 0, 0
dif_sum_count, dif_mul_count = 0, 0


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        return value, round(elapsed_time, 5)
    return wrapper_timer


def new_energy_treshold(energy_treshold, frame_energy):
    sigma_old, sigma_new = numpy.std(energy_treshold), numpy.std(frame_energy)
    if sigma_new > 0:
        print(sigma_new)
        print(sigma_old)
    try:
        sigma_div = float(sigma_new / sigma_old)
    except RuntimeWarning:
        sigma_div = 1
    except ZeroDivisionError:
        sigma_div = 1
    if sigma_div >= 1.25:
        p = 0.25
    elif 1.1 <= sigma_div <= 1.25:
        p = 0.2
    elif 1 <= sigma_div <= 1.1:
        p = 0.15
    else:
        p = 0.1
    return (1 - p) * energy_treshold + p * frame_energy


def vad_led(wav_data, fs, number_of_points, frame_size=0.02, k=2, p=0.1):
    duration = number_of_points / fs
    number_of_frames = int(numpy.ceil(duration / frame_size))
    points_in_frame = int(numpy.ceil(number_of_points / number_of_frames))
    wav_data = numpy.pad(wav_data, (0, number_of_frames * points_in_frame - number_of_points))
    vad_output = numpy.zeros_like(wav_data)

    # creation of the frames
    frames = numpy.reshape(wav_data, (number_of_frames, points_in_frame))

    # calculate the energy of each frame
    frame_energy = [short_term_energy(frame) for frame in frames]

    # 100ms assumed to be silence
    v = int(numpy.ceil(0.1 / frame_size))
    # calc the initial energy threshold
    energy_threshold = numpy.sum([frame_energy[m] for m in range(v)]) / v

    # LED: Linear Energy-Based Detector
    for index in range(v, number_of_frames):
        if frame_energy[index] > k * energy_threshold:
            # ACTIVE FRAME
            vad_output[index * points_in_frame: index * points_in_frame + points_in_frame].fill(1)
        else:
            # INACTIVE FRAME
            energy_threshold = (1 - p) * energy_threshold + p * frame_energy[index]
    return vad_output[0: number_of_points]


def short_term_energy(v):
    """
    :param v: array
    :return: energy=sigma(x^2)/n
    """
    return numpy.sum([numpy.abs(x)**2 for x in v]) / len(v)


def spectral_flatness_measure(x):
    """
    :param x: fft array absolute
    :return:  SFM=10log(geo_mean/mean)
    """
    return 10*numpy.log10(x.prod()**(1 / len(x)) / numpy.mean(x))


@timer
def dftmtx_mul_x(x):
    return numpy.dot(x, dftmtx(len(x)))


@timer
def fft(x):
    return numpy.fft.fft(x)


@timer
def cross_correlation_fft(signal_a, signal_b):
    n = len(signal_a) + len(signal_b) - 1
    _a = numpy.pad(signal_a, (0, int(n - len(signal_a))))
    _a = numpy.fft.fft(_a)
    _b = numpy.pad(signal_b, (0, int(n - len(signal_b))))
    _b = numpy.conj(numpy.fft.fft(_b))
    return numpy.real(numpy.fft.fftshift(numpy.fft.ifft(_a * _b)))


@timer
def cross_correlation(signal_a, signal_b):
    n = len(signal_a) + len(signal_b) - 1
    _b = signal_b[::-1]
    _a = numpy.pad(signal_a, (0, int(n - len(signal_a))))
    _b = numpy.pad(_b, (0, int(n - len(signal_b))))
    output = numpy.zeros_like(_a)
    for i in range(n):
        for j in range(i + 1):
            output[i] = output[i] + _a[j] * _b[i-j]
    return output


def cyclic_conv_zero_padding(signal_a, signal_b):
    _length = int(len(signal_a) + len(signal_b) - 1)
    _a = numpy.pad(signal_a, (0, _length - len(signal_a)))
    _b = numpy.pad(signal_b, (0, _length - len(signal_b)))
    output = numpy.fft.fft(_a) * numpy.fft.fft(_b)
    return numpy.fft.ifft(output)


def dftmtx(n):
    return numpy.fft.fft(numpy.eye(n))


def dit_fft_radix2(x):
    global dit_sum_count, dit_mul_count
    x = x.astype(numpy.complex_)
    p = numpy.ceil(numpy.log2(len(x)))
    n = int(pow(2, p))
    x = numpy.pad(x, (0, int(n - len(x))))
    y = numpy.asarray([x[i] for i in range(n)])
    if n == 1:
        return x
    else:
        twiddle = numpy.exp((-2J * numpy.pi) / n, dtype=numpy.complex_)
        w = 1
        x_even = dit_fft_radix2(x[0:n-1:2])
        x_odd = dit_fft_radix2(x[1:n:2])
        for index in range(0, int(n / 2)):
            y[index] = x_even[index] + w * x_odd[index]
            y[int(index + (n / 2))] = x_even[index] - w * x_odd[index]
            w = w * twiddle
            dit_sum_count = dit_sum_count + 2
            dit_mul_count = dit_mul_count + 1
        return y


def dif_fft_radix2(x):
    global dif_sum_count, dif_mul_count
    x = x.astype(numpy.complex_)
    log2length = int(numpy.ceil(numpy.log2(len(x))))
    x = numpy.pad(x, (0, 2**log2length - len(x)))
    twiddle = numpy.exp(-2J * numpy.pi * numpy.arange(0, 0.5, 1. / len(x), dtype=numpy.complex_))
    b_p = 1
    nvar_p = x.size
    twiddle_step_size = 1
    for _ in range(0,  log2length):
        nvar_pp = int(nvar_p/2)
        base_e = 0
        for _ in range(0,  b_p):
            base_o = int(base_e+nvar_pp)
            for nvar in range(0,  nvar_pp):
                evar = x[base_e + nvar] + x[base_o + nvar]
                dif_sum_count = dif_sum_count + 1
                if nvar == 0:
                    ovar = x[base_e + nvar] - x[base_o + nvar]
                else:
                    twiddle_factor = nvar*twiddle_step_size
                    ovar = (x[base_e + nvar] - x[base_o + nvar]) * twiddle[twiddle_factor]
                x[base_e + nvar] = evar
                x[base_o + nvar] = ovar
                dif_sum_count = dif_sum_count + 1
                dif_mul_count = dif_mul_count + 1
            base_e = int(base_e+nvar_p)
        b_p = int(b_p*2)
        nvar_p = int(nvar_p/2)
        twiddle_step_size = 2*twiddle_step_size
    return bit_reverse_vector_order(x, log2length)


def bit_reverse_vector_order(x, log2length):
    length = int(len(x))
    muplus = int((log2length+1)/2)
    mvar = 1
    reverse = numpy.zeros(length, dtype=int)
    upper_range = muplus+1
    for _ in numpy.arange(1, upper_range):
        for kvar in numpy.arange(0, mvar):
            tvar = 2*reverse[kvar]
            reverse[kvar] = tvar
            reverse[kvar+mvar] = tvar+1
        mvar = mvar+mvar
    if log2length & 0x01:
        mvar = mvar/2
    for qvar in numpy.arange(1, mvar):
        nprime = qvar-mvar
        rprimeprime = reverse[int(qvar)]*mvar
        for pvar in numpy.arange(0, reverse[int(qvar)]):
            nprime = int(nprime+mvar)
            rprime = int(rprimeprime+reverse[pvar])
            temp = x[nprime]
            x[nprime] = x[rprime]
            x[rprime] = temp
    return x


def filter_using_utils(x, h):
    n = int(pow(2, numpy.ceil(numpy.log2(len(x)))))
    x = numpy.pad(x, (0, int(n - len(x))))
    h = numpy.pad(h, (0, int(n - len(h))))
    output = dif_fft_radix2(x) * dit_fft_radix2(h)
    return numpy.conj(numpy.fft.fft(numpy.conj(output))/len(output))


def random_vector(n):
    return numpy.random.rand(n)


def regular_fft_ifft(x):
    return numpy.fft.ifft(numpy.fft.fft(x))


def conj_fft_ifft(x):
    return numpy.conj(numpy.fft.fft(numpy.conj(numpy.fft.fft(x)))) / len(x)


def conj_dft_idft_with_dftmtx(x):
    y1, t = dftmtx_mul_x(x)
    y1 = numpy.conj(y1)
    y1, t = dftmtx_mul_x(y1)
    return numpy.conj(y1) / len(x)


def get_plt():
    return plt


def find_freq_index(freq, f_vector):
    f_vector = numpy.asarray(f_vector)
    index = (numpy.abs(f_vector - freq)).argmin()
    return index


def get_plot(*args, title='', title_x='', title_y='', limx=None, limy=None, save_name=None):
    fig = plt.figure()
    for arg in args[1:]:
        plt.plot(args[0], arg)
    plt.title(title)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    if limx is not None:
        plt.xlim(limx)
    if limy is not None:
        plt.ylim(limy)
    if save_name:
        plt.savefig(save_name)
    return fig


def get_stem(*args, title='', title_x='', title_y='', limx=None, limy=None, save_name=None):
    fig = plt.figure()
    for arg in args[1:]:
        plt.stem(args[0], arg)
    plt.title(title)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    if limx is not None:
        plt.xlim(limx)
    if limy is not None:
        plt.ylim(limy)
    if save_name:
        plt.savefig(save_name)
    return fig
