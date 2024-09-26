from utills import *
import os
from settings import *
import pandas as pd
from tabulate import tabulate
def process_denoising(input_file, output_file, window_size=2048, hop_size=1024, threshold_factor=1.5):
    sample_rate, noisy_audio = wavfile.read(input_file)
    # Normalize the audio
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))

    # Perform block thresholding
    denoised_audio = block_thresholding(noisy_audio, sample_rate, window_size, hop_size, threshold_factor)

    # Normalize the denoised audio to prevent clipping
    denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))

    # Save the denoised audio
    wavfile.write(output_file, sample_rate, np.int16(denoised_audio * 32767))
    print(f"Denoised audio saved to {output_file}")
    return denoised_audio, sample_rate



# create main function
if __name__ == "__main__":

    if generate_noisy_data:
        process_clean_audio_with_noise(clean_data_path,noise_data_path, noisy_folder_path, noise_factor=0.005)
    
    results = []
    # Load the clean audio and noisy audio
    for file_name in os.listdir(noisy_folder_path):
        if file_name.endswith(".wav"):
            print(f"Processing file: {file_name}")
            noisy_input_path = os.path.join(noisy_folder_path, file_name)
            denoised_output_path = os.path.join(denoised_folder_path, "denoised_" + file_name)
            clean_audio_path = os.path.join(clean_data_path, file_name.replace("noised_", ""))
            denoised_audio, sample_rate = process_denoising(noisy_input_path, denoised_output_path, window_size=window_size, hop_size=hop_size, threshold_factor=1.5)
            _ , clean_audio = load_audio(clean_audio_path)
            _ ,  noisy_audio = load_audio(noisy_input_path)
            result = evaluate_denoising(clean_audio, noisy_audio, denoised_audio, sample_rate)
            results.append({
                'File': file_name,
                'SNR Noisy': result['SNR'][0],
                'SNR Denoised': result['SNR'][1],
                'PESQ Noisy': result['PESQ'][0],
                'PESQ Denoised': result['PESQ'][1],
                'STOI Noisy': result['STOI'][0],
                'STOI Denoised': result['STOI'][1],
                'MSE Noisy': result['MSE'][0],
                'MSE Denoised': result['MSE'][1],
                'PSNR Noisy': result['PSNR'][0],
                'PSNR Denoised': result['PSNR'][1],
            })
            plot_audio_signals(file_name,clean_audio, noisy_audio, denoised_audio, sample_rate)
    results_df = pd.DataFrame(results)
    # Calculate averages
    # Calculate average results
    avg_results = results_df.mean(numeric_only=True)  # Calculate mean for numeric columns
    avg_results['File'] = 'Average'  # Add label for average row

    # Append average results to the DataFrame
    results_df = pd.concat([results_df, avg_results.to_frame().T], ignore_index=True)

    print("\nDenoising Evaluation Results (Formatted):")
    print(tabulate(results_df, headers='keys', tablefmt='psql', showindex=False))
    # get it i csv
    results_df.to_csv('evaluations_results.csv', index=False)