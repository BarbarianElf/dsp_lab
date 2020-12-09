"""
utils lab 4 for DSP LAB

Created on December 8th 2020

@authors: Niv Ben Ami & Ziv Zango
"""
import numpy


def power_to_db(magnitude, epsilon=1e-10, top_db=80):
    """
    Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(magnitude)`` in a numerically
    stable way.
    in case magnitude is lower than epsilon (default 1e-10) computing with epsilon

    :return: Spectrogram in dB
    """

    db = 10 * numpy.log10(numpy.maximum(epsilon, magnitude))
    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        db = numpy.maximum(db, db.max() - top_db)
    return db


def freq_to_mel(freq):
    """
    convert frequency to MEL-scale
    """
    return 2595 * numpy.log10(1 + freq / 700)


def mel_to_freq(mels):
    """
    convert MEL-scale to frequency
    """
    return 700 * (10**(mels / 2595) - 1)


def pre_emphasis(signal, alpha=0.97):
    """
    Amplified high frequencies / Attenuator low frequencies

    :return: filtered signal
    """
    return numpy.append(signal[0], signal[1:] - alpha * signal[:-1])


def framing(signal, fs, frame_size=0.016, overlapping=0.5, **kwargs):
    """
    segment the input signal into frames of frame_size (default is 16ms)
    with overlap (default 50%) of the frame size

    :return: signal's frames
    """
    if overlapping < 0 or overlapping >= 1:
        raise ValueError('overlapping must be between 0 to 1 not including 1')
    number_of_points = len(signal)
    points_in_frame = int(numpy.ceil(frame_size * fs))
    # Number of frames include overlapping
    frame_step = int(numpy.ceil(points_in_frame * (1 - overlapping)))
    number_of_frames = int(numpy.ceil((number_of_points - points_in_frame) / frame_step)) + 1
    # Zero padding for the last frame
    framed_signal = numpy.pad(signal, (0, (number_of_frames * frame_step + frame_step - number_of_points)))
    # Creating the shape of the signal divided to frames (including overlapping)
    indices = numpy.lib.stride_tricks.as_strided(numpy.arange(len(framed_signal)),
                                                 (number_of_frames, points_in_frame),
                                                 (frame_step * 4, 4))
    # Frames creation
    return framed_signal[indices.astype(numpy.int32)]


def hamming_window(length):
    """
    The Hamming window is a taper formed by using a weighted cosine.

    :return: Hamming window.
    """
    return numpy.array([(0.54 - 0.46 * numpy.cos(2 * numpy.pi * k / (length - 1))) for k in range(length)])


def fft_points(signal):
    """
    find the nearest power of 2 of the signal

    :return: Points of fft
    """
    samples = signal.shape[signal.ndim - 1]
    p = numpy.log2(samples)
    if p.is_integer():
        p += 1
    else:
        p = numpy.ceil(p)
    return int(pow(2, p))


def power_spectrum(signal, points_of_fft=1024):
    """
    Perform fft with points_of_fft (default 1024),
    takes the absolute non-negative frequencies and power by 2
    `` P = |FFT(signal)|^2 / points_of_fft ``

    :return: Power spectrum of the signal
    """
    return (1 / points_of_fft) * (numpy.absolute(numpy.fft.rfft(signal, points_of_fft))) ** 2


def mel_filters_bank(fs, points_of_fft, mel_filters=40, normalized=False, **kwargs):
    """
    Computing filters bank points with freq_to_mel from 0 to fs/2 (max) the
    number of points is the non-negative frequencies points (points_of_fft/2 +1)
    and Construct the filters as triangular filters.

    number of the filters is mel_filters (default and typically 40)

    normalized the filters (default False) which divide the triangular MEL
    weights by the width of the MEL band (also called area normalization)

    :return: Mel-Filters Bank
    """
    low_mel = freq_to_mel(0)
    high_mel = freq_to_mel(fs / 2)
    mel_points = numpy.linspace(low_mel, high_mel, num=mel_filters + 2)
    freqs = mel_to_freq(mel_points)

    h = numpy.floor((points_of_fft + 1) / fs * freqs).astype(int)

    filters = numpy.zeros((mel_filters, int(points_of_fft / 2 + 1)))
    # filters creation
    for m in range(1, mel_filters + 1):
        m_left = h[m - 1]
        m_center = h[m]
        m_right = h[m + 1]

        for k in range(m_left, m_center):
            filters[m - 1, k] = (k - h[m - 1]) / (h[m] - h[m - 1])

        for k in range(m_center, m_right):
            filters[m - 1, k] = (h[m + 1] - k) / (h[m + 1] - h[m])

    if normalized:
        enorm = 2.0 / (freqs[2:mel_filters + 2] - freqs[:mel_filters])
        filters *= enorm[:, numpy.newaxis]

    return filters


def dct_filters(filter_len, dct_filters_num=12, **kwargs):
    """
    Calculate DCT type 2 orthogonal filters,
    number of filters is dct_filters_num (default 12)
    filter length is filter_len

    :return: Discrete Cosine Transform filters coefficients
    """
    filters = numpy.empty((dct_filters_num, filter_len))
    filters[0, :] = 2.0 / numpy.sqrt(4 * filter_len)

    samples = numpy.arange(1, 2 * filter_len, 2) / (2.0 * filter_len)

    for i in range(1, dct_filters_num):
        filters[i, :] = 2 * numpy.cos(numpy.pi * i * samples) * numpy.sqrt(1.0 / (2.0 * filter_len))

    return filters
