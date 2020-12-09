"""
lab 4 for DSP LAB

Created on November 27th 2020

@authors: Niv Ben Ami & Ziv Zango
"""

import warnings
import numpy
import librosa
import os

import LAB4.mfcc_utils as utils
import lab_utils as lu


def mel_spectrogram(signal, fs, **kwargs):
    """
    Compute a mel-scaled spectrogram of the ``signal``
    with sample rate of ``fs``

    Optional parameters
    -------------------
    mel_filters : int
        number of mel filters bank (default 40)
    normalized : boolean (default False)
        If ``normalized`` provided as True, mel filters will be in Slaney-style
    frame_size : float
        size of the frame is seconds (default 0.016)
    overlapping : float (between 0 to 1)
        overlap as percentage of the frame (default 0.5 i.e. 50%)

    :return: mel-scaled spectrogram (magnitude)
    :return: mel filters
    """
    frames = utils.framing(signal, fs, **kwargs)
    w = utils.hamming_window(frames.shape[frames.ndim - 1])
    frames = frames.astype(w.dtype)

    # frames filtered with hamming window
    frames *= w
    fft_points = utils.fft_points(frames)
    frames_power = utils.power_spectrum(frames, fft_points)
    filters = utils.mel_filters_bank(fs, fft_points, **kwargs)

    # frames power filtered with mel filters bank
    frames_power_filtered = numpy.dot(filters, numpy.transpose(frames_power))
    return frames_power_filtered, filters


def mf_cepstral_coefficients(signal, fs, pre_emphasis=True, **kwargs):
    """
    Compute a mel-frequency cepstral coefficients ``signal``
    with sample rate of ``fs`` using the function mel_spectrogram

    Optional parameters
    -------------------
    pre_emphasis : boolean (default True)
        If ``normalized`` provided as False, pre_emphasis wont be performed
    dct_filters_num : int
        number of cepsral coefficients (default 12)

    including optional parameters of mel_spectrogram function
    see also mel_spectrogram

    :return: Mel-Frequency Cepstral Coefficients
    """
    if pre_emphasis:
        signal = utils.pre_emphasis(signal)
    frames_power, filters = mel_spectrogram(signal, fs, **kwargs)
    frames_db = utils.power_to_db(frames_power)

    # frames in db filtered with dct coefficients
    cepstral_coeff = numpy.dot(utils.dct_filters(frames_db.shape[0], **kwargs), frames_db)
    return cepstral_coeff


if __name__ == "__main__":
    plt = lu.get_plt()
    plt.rcParams.update({'figure.max_open_warning': 0})
    os.makedirs("results", exist_ok=True)

    # insert to list all the files with wav extension
    files = []
    for file in os.listdir("."):
        if file.endswith(".wav"):
            files.append(file)

    # LAB4
    figure_num = 0
    for file in files:
        try:
            name = file.split('_')[1]
            name = name.split('.')[0]
        except Exception as Err:
            warnings.warn("taking full file name instead of name, file name not as expected")
            name = file.split('.')[0]

        sound_data, frequency_sampled = librosa.core.load(file, sr=8000)

        spectrogram, filters_bank = mel_spectrogram(sound_data,
                                                    frequency_sampled,
                                                    mel_filters=40,
                                                    normalized=True)

        mfcc = mf_cepstral_coefficients(sound_data,
                                        frequency_sampled,
                                        pre_emphasis=False,
                                        frame_size=0.016,
                                        overlapping=0.5,
                                        mel_filters=40,
                                        dct_filters_num=12,
                                        normalized=True)

        libros_s = librosa.feature.melspectrogram(sound_data, sr=frequency_sampled, n_fft=256, hop_length=64, n_mels=40)
        libros_mfcc = librosa.feature.mfcc(sound_data, sr=frequency_sampled, n_mfcc=12, hop_length=64)

        plt.figure(figure_num)
        plt.title(f'{name}\nAudio signal')
        plt.xlabel('Time(s)')
        plt.plot(numpy.linspace(0, len(sound_data) / frequency_sampled, num=len(sound_data)), sound_data)
        plt.grid(True)
        plt.savefig(f"results/{name}_Audio signal")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nSpectrogram (Magnitude)')
        plt.ylabel('Frequency(Hz)')
        plt.xlabel('Time(s)')
        plt.imshow(spectrogram,
                   cmap='jet',
                   aspect='auto',
                   origin='lower',
                   extent=[0, len(sound_data) / frequency_sampled, 0, frequency_sampled / 2])
        plt.savefig(f"results/{name}_Spectrogram_m")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nSpectrogram (dB)')
        plt.ylabel('Frequency(Hz)')
        plt.xlabel('Time(s)')
        plt.imshow(utils.power_to_db(spectrogram),
                   cmap='jet',
                   aspect='auto',
                   origin='lower',
                   extent=[0, len(sound_data) / frequency_sampled, 0, frequency_sampled / 2])
        plt.savefig(f"results/{name}_Spectrogram_db")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nMel-Frequency Cepstral Coefficients')
        plt.pcolormesh(mfcc)
        plt.savefig(f"results/{name}_MFCC")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nLIBROSA: Spectrogram (Magnitude)')
        plt.ylabel('Frequency(Hz)')
        plt.xlabel('Time(s)')
        plt.imshow(libros_s,
                   cmap='jet',
                   aspect='auto',
                   origin='lower',
                   extent=[0, len(sound_data) / frequency_sampled, 0, frequency_sampled / 2])
        plt.savefig(f"results/{name}_LIBROSA_Spectrogram_m")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nLIBROSA: Spectrogram (dB)')
        plt.ylabel('Frequency(Hz)')
        plt.xlabel('Time(s)')
        plt.imshow(librosa.power_to_db(libros_s),
                   cmap='jet',
                   aspect='auto',
                   origin='lower',
                   extent=[0, len(sound_data) / frequency_sampled, 0, frequency_sampled / 2])
        plt.savefig(f"results/{name}_LIBROSA_Spectrogram_db")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nLIBROSA: Mel-Frequency Cepstral Coefficients')
        plt.pcolormesh(libros_mfcc)
        plt.savefig(f"results/{name}_LIBROSA_mfcc")
        figure_num += 1

    try:
        plt.figure(figure_num, figsize=(10, 4))
        plt.title('MEL filters bank')
        plt.ylabel('Amplitude')
        plt.xlabel('Frequency(Hz)')
        for n in range(filters_bank.shape[0]):
            plt.plot(numpy.linspace(0, frequency_sampled / 2, filters_bank.shape[1]), filters_bank[n])
        plt.savefig(f"results/MEL filters bank")
    except Exception as Err:
        pass

    # Unmark the command blow to show ALL the figures
    # plt.show()
