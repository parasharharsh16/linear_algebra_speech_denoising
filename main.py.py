from utills import *
import os
from settings import *

if (generate_noisy_data == True):
    # Generate noisy signals
    noise_ratio = 0.3
    # loop on clean and noise waves from folders
    for wavfile in os.listdir(clean_data_path):
        for noisefile in os.listdir(noise_data_path):
            clean_wav, clean_sr = load_wav(clean_data_path+"/"+wavfile)
            noise_wav, noise_sr = load_wav(noise_data_path+"/"+noisefile)
            noisy_wav = generate_noisy_wave(clean_wav, noise_wav, noise_ratio)
            #save noisy wave
            #get noisy file name with combined clean and noise file names
            noisy_file = wavfile.replace(".wav", "") + "_" + noisefile.replace(".wav", "") + "_noisy.wav"
            save_wav(noisy_wav, noisy_folder_path+"/"+noisy_file, clean_sr)

# Load the each noisy wave and denoise it
for noisy_wav in os.listdir(noisy_folder_path):
    noisy_wave, sample_rate = load_wav(noisy_folder_path+"/"+noisy_wav)
    # Denoise the audio
    window_size = 1024
    hop_size = 512
    threshold = 0.1
    denoised_audio = denoise_audio(noisy_wave, sample_rate, window_size, hop_size, threshold)
    # Save the denoised audio
    denoised_file = noisy_wav.replace("noisy", "denoised")
    save_wav(denoised_audio, denoised_folder_path+"/"+denoised_file, sample_rate)
    #plot noisy vs denoised audio
    #get plot name by combination of clean and noise file names
    plot_name = noisy_wav.replace("noisy", "denoised").replace(".wav", "")
    plot_audio(noisy_wave, denoised_audio, sample_rate, plot_name)

    print("Denoised audio saved as: ", denoised_folder_path+"/"+denoised_file)
    print("Denoising completed for: ", noisy_wav)
    print("--------------------------------------------------")
