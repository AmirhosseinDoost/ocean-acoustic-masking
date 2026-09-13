import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from scipy.signal import butter, filtfilt, resample


def highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


def compute_bispectrum(x, sr, nfft=2048, max_freq=1500):
    freq_res = sr / nfft
    max_bin = int(max_freq / freq_res)
    N = len(x)
    n_blocks = N // nfft

    if n_blocks == 0:
        raise ValueError("Signal is too short for the given nfft.")

    x_blocks = x[:n_blocks * nfft].reshape(n_blocks, nfft)
    window = np.hanning(nfft)
    x_blocks = x_blocks * window

    X = np.fft.fft(x_blocks, axis=1)
    B = np.zeros((max_bin, max_bin), dtype=complex)

    for f1 in range(max_bin):
        for f2 in range(max_bin):
            if f1 + f2 < X.shape[1]:
                B[f1, f2] = np.mean(X[:, f1] * X[:, f2] * np.conj(X[:, f1 + f2]))

    freqs = np.arange(max_bin) * freq_res
    return np.abs(B), freqs


def run_deepwater_bispectrum_filtered(base_dir_ships, deepwater_file, code_dir):
    output_dir = os.path.join(code_dir, "Result_16_Deepwater_Bispectrum")
    os.makedirs(output_dir, exist_ok=True)

    sig_noise, sr_noise = sf.read(deepwater_file)
    if sig_noise.ndim > 1: sig_noise = sig_noise[:, 0]

    excel_path = os.path.join(base_dir_ships, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)
    target_files = df[df['Type'].str.lower() == 'passengers']['Filename'].dropna()
    target_path = os.path.join(base_dir_ships, target_files.iloc[0])
    sr_target, sig_target = wavfile.read(target_path)
    if sig_target.ndim > 1: sig_target = sig_target[:, 0]

    if sr_noise != sr_target:
        duration = min(len(sig_noise) / sr_noise, len(sig_target) / sr_target, 10.0)
        num_samples_target = int(duration * sr_noise)
        sig_target = resample(sig_target[:int(duration * sr_target)], num_samples_target)
        sig_noise = sig_noise[:num_samples_target]
        sr = sr_noise
    else:
        min_len = min(len(sig_noise), len(sig_target), int(10 * sr_noise))
        sig_noise = sig_noise[:min_len]
        sig_target = sig_target[:min_len]
        sr = sr_noise

    # اعمال فیلتر بالاگذر برای حذف DC Offset و فشار هیدرواستاتیک
    sig_n = highpass_filter(sig_noise.astype(float), cutoff=20.0, fs=sr)
    sig_t = highpass_filter(sig_target.astype(float), cutoff=20.0, fs=sr)

    sig_n /= (np.max(np.abs(sig_n)) + 1e-10)
    sig_t /= (np.max(np.abs(sig_t)) + 1e-10)

    test_ratio = 0.10
    sig_mix = (test_ratio * sig_t) + ((1.0 - test_ratio) * sig_n)

    print("Computing Filtered Bispectrum for Pure Deepwater Noise...")
    B_noise, freqs = compute_bispectrum(sig_n, sr, nfft=2048, max_freq=1000)

    print("Computing Filtered Bispectrum for Pure Vessel...")
    B_target, _ = compute_bispectrum(sig_t, sr, nfft=2048, max_freq=1000)

    print(f"Computing Filtered Bispectrum for Mixed Signal (SNR: {test_ratio * 100}%)...")
    B_mix, _ = compute_bispectrum(sig_mix, sr, nfft=2048, max_freq=1000)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style="white")

    def plot_contour(ax, B_matrix, title):
        B_norm = B_matrix / (np.max(B_matrix) + 1e-15)
        X, Y = np.meshgrid(freqs, freqs)
        contour = ax.contourf(X, Y, B_norm, levels=30, cmap='magma')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency f1 (Hz)')
        ax.set_ylabel('Frequency f2 (Hz)')
        return contour

    plot_contour(axes[0], B_noise, 'Pure Deepwater Noise (Filtered)\nExpected: Absolute Zero Coupling')
    plot_contour(axes[1], B_target, 'Pure Vessel Signature\nExpected: High Phase Coupling')
    plot_contour(axes[2], B_mix, f'Mixed Signal ({int(test_ratio * 100)}% Vessel)\nDeepwater Exposure Test')

    plt.tight_layout()
    output_plot = os.path.join(output_dir, "Deepwater_Bispectrum_Filtered.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"\nDone! Filtered Deepwater Bispectrum saved to: {output_plot}")


base_ships = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
deep_file = r"D:\Academic Projects\Passive\Dataset\ADEON_Data\deepwater_calm.flac"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"

run_deepwater_bispectrum_filtered(base_ships, deep_file, code_directory)