# Audio Denoising using STFT and Linear Filters

## Description
This project focuses on denoising audio signals using Short-Time Fourier Transform (STFT) and linear filtering techniques. It implements methods for adding noise to clean audio, applying adaptive filtering, and evaluating the performance of the denoised audio.

## Project Setup and Usage Steps
### Installation
To install and run this project, follow these steps:

1. Clone the repository: ```git clone git@github.com:parasharharsh16/linear_algebra_speech_denoising.git ```
2. Navigate into the project directory: ```cd linear_algebra_speech_denoising```
3. Create a conda (use Miniconda or Anaconda) environment: ```conda create --prefix ./.venv python=3.9```
4. Activate the conda environment: ```conda activate ./.venv```
5. Install the required dependencies: ```pip install -r requirements.txt```

### Prepare Dataset
The dataset is already included in the repository in the `data` folder, which contains:
- **clean audio**: Folder with clean audio files.
- **noise audio**: Folder with noise audio files.

When the code is executed, it automatically creates noisy input by mixing the clean and noise audio files and stores them in a folder called `noisyaudio`.

### Run the Code
1. Adjust the parameters in `settings.py` according to your needs.
2. Run the main script: `python main.py`

### Results
- The `output/denoisedaudio` folder contains generated denoised audio files.
- The `output/plots` folder contains plots comparing cleay, noisy and denoised waveforms. 

## Evaluation Metrics
The project evaluates the quality of the denoised audio using:
- **SNR**: Signal-to-Noise Ratio.
- **PESQ**: Perceptual Evaluation of Speech Quality.
- **STOI**: Short-Time Objective Intelligibility.
- **MSE**: Mean Squared Error.
- **PSNR**: Peak Signal-to-Noise Ratio.

## Results

| File                   | SNR Noisy | SNR Denoised | PESQ Noisy | PESQ Denoised | STOI Noisy | STOI Denoised | MSE Noisy | MSE Denoised | PSNR Noisy | PSNR Denoised |
|------------------------|-----------|--------------|------------|---------------|------------|----------------|-----------|--------------|------------|----------------|
| noised_p228_003.wav    | 3.32     | -32.25       | 1.17       | 1.13          | 0.75       | 0.73           | 999.01    | 3598151.84    | 56.43      | 20.87          |
| noised_p228_004.wav    | 5.97     | -30.80       | 1.17       | 1.71          | 0.77       | 0.77           | 672.21    | 3195769.51    | 56.13      | 19.36          |
| noised_p257_004.wav    | 2.81     | -34.94       | 1.32       | 1.31          | 0.95       | 0.94           | 1111.79   | 6627045.69    | 57.71      | 19.96          |
| noised_p257_002.wav    | 3.04     | -27.90       | 1.39       | 1.34          | 0.95       | 0.95           | 1162.97   | 1441862.45    | 49.89      | 18.96          |
| noised_p257_003.wav    | 1.17     | -32.67       | 3.09       | 1.72          | 0.95       | 0.93           | 1461.65   | 3537393.54    | 51.00      | 17.16          |
| noised_p228_005.wav    | 3.67     | -32.85       | 1.30       | 1.22          | 0.82       | 0.80           | 751.38    | 3378909.27    | 54.74      | 18.21          |
| **Average**            | **3.33** | **-31.90**   | **1.57**   | **1.41**      | **0.86**   | **0.85**       | **1026.50**| **3629855.38**| **54.32**  | **19.09**      |


## Results Interpretation

- **SNR (Signal-to-Noise Ratio):** The average SNR of the noisy audio is about **3.33 dB**. After denoising, it drops to **-31.90 dB**, indicating that while noise is present, the cleaning process did not significantly improve the signal quality.

- **PESQ (Perceptual Evaluation of Speech Quality):** The average PESQ score decreased from **1.57** in the noisy audio to **1.41** in the cleaned audio. This suggests that while there was a slight improvement in quality, some distortion remains noticeable.

- **STOI (Short-Time Objective Intelligibility):** The average STOI score changed slightly from **0.86** to **0.85**, indicating that some speech clarity was preserved despite the noise.

- **MSE (Mean Squared Error):** The MSE increased from **1026.50** for the noisy audio to **3629855.38** for the denoised audio, suggesting that the denoising process introduced significant errors.

- **PSNR (Peak Signal-to-Noise Ratio):** The average PSNR dropped from **54.32 dB** to **19.09 dB**. This implies that while some noise was removed, the cleaning process may have introduced new issues that affected the overall audio quality.

### Result Images

Below is a plot showing the clean, noisy, and denoised audio signals for one audio file:

![Clean, Noisy, and Denoised Audio](https://github.com/parasharharsh16/linear_algebra_speech_denoising/blob/main/output/plots/noised_p228_005.wav_audio_signals.png)
*Plot of Clean, Noisy, and Denoised Audio*

## Team Members

- Harsh Parashar (M22AIE210)
- Prateek Singhal (M22AIE215)
- Prabha Sharma (M22AIE224)
- Ashish Rawat (M22AIE201)
- Amit Pawar (M22AIE202)
