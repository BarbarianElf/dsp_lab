"""
lab 4 for DSP LAB

Created on November 27th 2020

@authors: Niv Ben Ami & Ziv Zango
"""

import numpy
import librosa
import os

import LAB4.mfcc_utils as utils
import lab_utils as lu


def mel_spectrogram(signal, fs, **kwargs):
    frames = utils.framing(signal, fs, **kwargs)
    w = utils.hamming_window(frames.shape[frames.ndim - 1])
    frames = frames.astype(w.dtype)
    frames *= w
    fft_points = utils.fft_points(frames)
    frames_power = utils.power_spectrum(frames, fft_points)
    filters = utils.mel_filters_bank(fs, fft_points, **kwargs)
    frames_power_filtered = numpy.dot(filters, numpy.transpose(frames_power))
    frames_db_filtered = utils.power_to_db(frames_power_filtered)
    return frames_db_filtered, filters


def mf_cepstral_coefficients(signal, fs, pre_emphasis=True, **kwargs):
    if pre_emphasis:
        signal = utils.pre_emphasis(signal)
    frames_db, filters_bank = mel_spectrogram(signal, fs, **kwargs)

    cepstral_coeff = numpy.dot(utils.dct_filters(frames_db.shape[0], **kwargs), frames_db)
    return cepstral_coeff


if __name__ == "__main__":
    plt = lu.get_plt()
    os.makedirs("results", exist_ok=True)
    files = ["shalom_record_ziv.wav", "shalom_record_niv.wav"]
    # LAB4
    figure_num = 0
    for file in files:
        name = file.split('_')[2]
        name = name.split('.')[0]
        sound_data, frequency_sampled = librosa.core.load(file, sr=8000)

        spectrogram, filters_bank = mel_spectrogram(sound_data,
                                                    frequency_sampled,
                                                    mel_filters=40,
                                                    normalized=True)

        mfcc = mf_cepstral_coefficients(sound_data,
                                        frequency_sampled,
                                        pre_emphasis=False,
                                        overlapping=0.5,
                                        mel_filters=40,
                                        dct_filters_num=12,
                                        normalized=True)

        librosa_s = librosa.feature.melspectrogram(sound_data, sr=frequency_sampled, n_fft=256, hop_length=64, n_mels=40)
        librosa_s = librosa.power_to_db(librosa_s)
        librosa_mfcc = librosa.feature.mfcc(sound_data, sr=frequency_sampled, n_mfcc=12, hop_length=64)

        plt.figure(figure_num)
        plt.title(f'{name}\nAudio signal')
        plt.xlabel('Time(s)')
        plt.plot(numpy.linspace(0, len(sound_data) / frequency_sampled, num=len(sound_data)), sound_data)
        plt.grid(True)
        plt.savefig(f"results/{name}_Audio signal")
        figure_num += 1

        plt.figure(figure_num, figsize=(10, 4))
        plt.title(f'{name}\nMEL filters bank')
        plt.ylabel('Amplitude')
        plt.xlabel('Frequency(Hz)')
        for n in range(filters_bank.shape[0]):
            plt.plot(numpy.linspace(0, frequency_sampled / 2, filters_bank.shape[1]), filters_bank[n])
        plt.savefig(f"results/{name}_MEL filters bank")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nSpectrogram')
        plt.ylabel('Frequency(Hz)')
        plt.xlabel('Time(s)')
        plt.imshow(spectrogram,
                   cmap='jet',
                   aspect='auto',
                   origin='lower',
                   extent=[0, len(sound_data) / frequency_sampled, 0, frequency_sampled / 2])
        plt.savefig(f"results/{name}_Spectrogram")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nMel-Frequency Cepstral Coefficients')
        plt.pcolormesh(mfcc)
        plt.savefig(f"results/{name}_MFCC")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nLIBROSA: Spectrogram')
        plt.ylabel('Frequency(Hz)')
        plt.xlabel('Time(s)')
        plt.imshow(librosa_s,
                   cmap='jet',
                   aspect='auto',
                   origin='lower',
                   extent=[0, len(sound_data) / frequency_sampled, 0, frequency_sampled / 2])
        plt.savefig(f"results/{name}_LIBROSA_Spectrogram")
        figure_num += 1

        plt.figure(figure_num)
        plt.title(f'{name}\nLIBROSA: Mel-Frequency Cepstral Coefficients')
        plt.pcolormesh(librosa_mfcc)
        plt.savefig(f"results/{name}_LIBROSA_mfcc")
        figure_num += 1

    # Unmark the command blow to show ALL the figures
    plt.show()
