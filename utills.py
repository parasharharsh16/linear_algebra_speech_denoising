import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft, lfilter, resample
from settings import *
from pesq import pesq
from pystoi import stoi

def load_audio(file_path):
    sample_rate, audio = wavfile.read(file_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return sample_rate, audio

def resample_audio(audio, orig_sr, target_sr):
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled_audio = resample(audio, num_samples)
    return resampled_audio

def add_noise_from_file(clean_audio, clean_sample_rate, noise_file_path, noise_factor=noise_factor, amplification_factor=amplification_factor):
    noise_sample_rate, noise_audio = wavfile.read(noise_file_path)
    
    if noise_audio.ndim > 1:
        noise_audio = noise_audio.mean(axis=1)
    
    if noise_sample_rate != clean_sample_rate:
        noise_audio = resample_audio(noise_audio, noise_sample_rate, clean_sample_rate)
    
    if len(noise_audio) < len(clean_audio):
        noise_audio = np.tile(noise_audio, int(np.ceil(len(clean_audio) / len(noise_audio))))
    
    noise_audio = noise_audio[:len(clean_audio)]
    
    if np.max(np.abs(noise_audio)) == 0:
        print("Warning: Noise is silent or blank.")
        return clean_audio
    
    noise_audio = noise_audio / np.max(np.abs(noise_audio))
    amplified_noise = noise_audio * amplification_factor
    scaled_noise = amplified_noise * noise_factor * np.max(np.abs(clean_audio))
    noised_audio = clean_audio + scaled_noise
    noised_audio = noised_audio / np.max(np.abs(noised_audio))
    
    return noised_audio

def save_audio(file_path, sample_rate, audio):
    audio_scaled = np.int16(audio * 32767)
    wavfile.write(file_path, sample_rate, audio_scaled)

def process_clean_audio_with_noise(input_folder, noise_folder, output_folder, noise_factor=noise_factor, amplification_factor=amplification_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    noise_files = [f for f in os.listdir(noise_folder) if f.endswith('.wav')]
    if not noise_files:
        raise FileNotFoundError(f"No noise files found in {noise_folder}")
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            sample_rate, clean_audio = load_audio(file_path)
            
            noise_file_path = os.path.join(noise_folder, np.random.choice(noise_files))
            noised_audio = add_noise_from_file(clean_audio, sample_rate, noise_file_path, noise_factor, amplification_factor)
            
            output_path = os.path.join(output_folder, f'noised_{filename}')
            save_audio(output_path, sample_rate, noised_audio)
            print(f'Saved noised audio: {output_path}')

def adaptive_thresholding(magnitude_noisy, noise_floor, factor=1.5):
    threshold = factor * noise_floor
    return np.where(magnitude_noisy > threshold, magnitude_noisy, 0)

def wiener_filter(magnitude_noisy, noise_estimate):
    noise_power = noise_estimate ** 2
    signal_power = magnitude_noisy ** 2
    wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
    return magnitude_noisy * wiener_gain

def apply_smoothing(audio, window_size=5):
    kernel = np.ones(window_size) / window_size
    return lfilter(kernel, 1, audio)

def block_thresholding(noisy_audio, sample_rate, window_size, hop_size, threshold_factor):
    f, t, stft_noisy = stft(noisy_audio, fs=sample_rate, nperseg=window_size, noverlap=hop_size)
    magnitude_noisy = np.abs(stft_noisy)

    noise_floor = np.median(magnitude_noisy, axis=1, keepdims=True)
    denoised_magnitude = adaptive_thresholding(magnitude_noisy, noise_floor, threshold_factor)
    denoised_magnitude = wiener_filter(denoised_magnitude, noise_floor)

    denoised_stft = denoised_magnitude * np.exp(1j * np.angle(stft_noisy))
    _, denoised_audio = istft(denoised_stft, fs=sample_rate, nperseg=window_size, noverlap=hop_size)
    denoised_audio = apply_smoothing(denoised_audio)

    return denoised_audio

def downsample_audio(audio, original_sr, target_sr):
    num_samples = int(len(audio) * target_sr / original_sr)
    return resample(audio, num_samples)

def evaluate_denoising(clean_audio, noisy_audio, denoised_audio, sample_rate):
    min_length = min(len(clean_audio), len(noisy_audio), len(denoised_audio))
    clean_audio = clean_audio[:min_length]
    noisy_audio = noisy_audio[:min_length]
    denoised_audio = denoised_audio[:min_length]

    target_sr = 16000
    clean_audio_ds = downsample_audio(clean_audio, sample_rate, target_sr)
    noisy_audio_ds = downsample_audio(noisy_audio, sample_rate, target_sr)
    denoised_audio_ds = downsample_audio(denoised_audio, sample_rate, target_sr)

    snr_noisy = 10 * np.log10(np.sum(clean_audio ** 2) / np.sum((clean_audio - noisy_audio) ** 2))
    snr_denoised = 10 * np.log10(np.sum(clean_audio ** 2) / np.sum((clean_audio - denoised_audio) ** 2))

    pesq_noisy = pesq(target_sr, noisy_audio_ds, clean_audio_ds, 'wb')
    pesq_denoised = pesq(target_sr, denoised_audio_ds, clean_audio_ds, 'wb')

    stoi_noisy = stoi(clean_audio_ds, noisy_audio_ds, target_sr, extended=False)
    stoi_denoised = stoi(clean_audio_ds, denoised_audio_ds, target_sr, extended=False)

    mse_noisy = np.mean((clean_audio - noisy_audio) ** 2)
    mse_denoised = np.mean((clean_audio - denoised_audio) ** 2)

    psnr_noisy = 10 * np.log10(np.max(clean_audio) ** 2 / mse_noisy) if mse_noisy > 0 else float('inf')
    psnr_denoised = 10 * np.log10(np.max(clean_audio) ** 2 / mse_denoised) if mse_denoised > 0 else float('inf')

    return {
        'SNR': (snr_noisy, snr_denoised),
        'PESQ': (pesq_noisy, pesq_denoised),
        'STOI': (stoi_noisy, stoi_denoised),
        'MSE': (mse_noisy, mse_denoised),
        'PSNR': (psnr_noisy, psnr_denoised)
    }

def plot_audio_signals(file_name, clean_audio, noisy_audio, denoised_audio, sample_rate):
    time_clean = np.arange(len(clean_audio)) / sample_rate
    time_noisy = np.arange(len(noisy_audio)) / sample_rate
    time_denoised = np.arange(len(denoised_audio)) / sample_rate

    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(time_clean, clean_audio, color='g')
    axs[0].set_title('Clean Audio Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid()

    axs[1].plot(time_noisy, noisy_audio, color='r')
    axs[1].set_title('Noisy Audio Signal')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid()

    axs[2].plot(time_denoised, denoised_audio, color='b')
    axs[2].set_title('Denoised Audio Signal')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid()

    plt.tight_layout()
    plt.savefig(f'{plot_folder_path}/{file_name}_audio_signals.png')
