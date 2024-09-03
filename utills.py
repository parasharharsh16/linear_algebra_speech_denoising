import numpy as np
import matplotlib.pyplot as plt
import scipy

# define function to load wav files (audio)
def load_wav(file_path):
    # load wav file
    wav = scipy.io.wavfile.read(file_path)
    # get sample rate
    sample_rate = wav[0]
    # get audio data
    audio = wav[1]
    return audio, sample_rate

def dft(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exp_matrix = np.exp(-2j * np.pi * k * n / N)
    return np.dot(exp_matrix, signal)

def stft(signal, window_size, hop_size, window_function=np.hamming):
    # Generate window function (e.g., Hamming or Hanning)
    window = window_function(window_size)
    
    # Number of frames
    num_frames = (len(signal) - window_size) // hop_size + 1
    
    # STFT matrix to hold the results
    stft_matrix = np.zeros((window_size, num_frames), dtype=complex)
    
    # Apply windowing and DFT to each frame
    for i in range(num_frames):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        
        # Windowed segment
        segment = signal[start_idx:end_idx] * window
        
        # Apply DFT to the segment
        stft_matrix[:, i] = dft(segment)
    
    return stft_matrix

def generate_test_signal(sample_rate, duration, freqs):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = sum(np.sin(2 * np.pi * freq * t) for freq in freqs)
    return signal

def plot_stft(stft_matrix, sample_rate, hop_size,window_size):
    time_bins = np.arange(stft_matrix.shape[1]) * hop_size / sample_rate
    freq_bins = np.fft.fftfreq(window_size, 1 / sample_rate)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time_bins, freq_bins[:window_size//2], np.abs(stft_matrix[:window_size//2, :]), shading='gouraud')
    plt.title('Short-Time Fourier Transform (STFT)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.show()

#Function to denoise audio using STFT
def denoise_audio(audio, sample_rate, window_size, hop_size, threshold):
    # Compute the STFT
    stft_matrix = stft(audio, window_size, hop_size)
    
    # Compute the magnitude spectrogram
    magnitude_spectrogram = np.abs(stft_matrix)
    
    # Apply thresholding
    magnitude_spectrogram[magnitude_spectrogram < threshold] = 0
    
    # Reconstruct the denoised audio
    denoised_stft = stft_matrix * (magnitude_spectrogram > 0)
    denoised_audio = np.real(np.sum(denoised_stft, axis=0))
    
    return denoised_audio

# visualize noised vs denoised audio
def plot_audio(audio, denoised_audio, sample_rate, plot_name):
    # Ensure audio and denoised_audio have the same length
    min_length = min(len(audio), len(denoised_audio))
    audio = audio[:min_length]
    denoised_audio = denoised_audio[:min_length]
    
    # Generate the time array with the same length as audio
    time = np.arange(min_length) / sample_rate
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, audio, label='Noised Audio')
    plt.plot(time, denoised_audio, label='Denoised Audio', linestyle='--')
    plt.title('Noised vs Denoised Audio')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig("plots/" + plot_name + ".png")
    plt.close()


# Generate noisy wave by mixing clean with noise
def generate_noisy_wave(clean_wave, noise_wave, noise_ratio):
    # Normalize the clean wave
    clean_wave = clean_wave / np.max(np.abs(clean_wave))
    
    # Normalize the noise wave
    noise_wave = noise_wave / np.max(np.abs(noise_wave))
    #Repeat the shorter array to match the length of the longer array
    if len(clean_wave) > len(noise_wave):
        noise_wave = np.tile(noise_wave, len(clean_wave) // len(noise_wave) + 1)
    else:
        clean_wave = np.tile(clean_wave, len(noise_wave) // len(clean_wave) + 1)
    #Truncate or pad the shorter array to match the length of the longer array.

    if len(clean_wave) > len(noise_wave):
        noise_wave = np.append(noise_wave, np.zeros(len(clean_wave) - len(noise_wave)))
    else:
        clean_wave = np.append(clean_wave, np.zeros(len(noise_wave) - len(clean_wave)))
    # Compute the noisy wave
    noisy_wave = clean_wave + noise_ratio * noise_wave
    
    return noisy_wave

#save noisy wave
def save_wav(wave, filename, sample_rate):
    wave = np.int16(wave * 32767)
    scipy.io.wavfile.write(filename, sample_rate, wave)