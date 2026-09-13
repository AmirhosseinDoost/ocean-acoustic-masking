import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns


def compute_bispectrum(x, sr, nfft=2048, max_freq=1500):
    """
    محاسبه بهینه طیف دوگانه فقط برای باندهای فرکانسی پایین (محدوده صدای مکانیکی)
    """
    freq_res = sr / nfft
    max_bin = int(max_freq / freq_res)

    N = len(x)
    n_blocks = N // nfft

    # حذف داده‌های اضافی انتهای سیگنال و شبکه‌بندی
    if n_blocks == 0:
        raise ValueError("Signal is too short for the given nfft.")

    x_blocks = x[:n_blocks * nfft].reshape(n_blocks, nfft)

    # اعمال پنجره برای کاهش نشت فرکانسی
    window = np.hanning(nfft)
    x_blocks = x_blocks * window

    X = np.fft.fft(x_blocks, axis=1)

    # ماتریس بای‌اسپکتروم
    B = np.zeros((max_bin, max_bin), dtype=complex)

    for f1 in range(max_bin):
        for f2 in range(max_bin):
            if f1 + f2 < X.shape[1]:
                # میانگین‌گیری روی تمام پنجره‌ها (Expected Value)
                B[f1, f2] = np.mean(X[:, f1] * X[:, f2] * np.conj(X[:, f1 + f2]))

    freqs = np.arange(max_bin) * freq_res
    return np.abs(B), freqs


def run_bispectrum_analysis(base_dir, code_dir):
    output_dir = os.path.join(code_dir, "Result_14_Bispectrum_Analysis")
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)

    # 1. بارگذاری یک نویز به شدت یکنواخت و گاوسی (مثلاً باد یا محیط آرام)
    noise_files = df[df['Type'].str.lower() == 'natural ambient noise']['Filename'].dropna()
    bg_noise_path = os.path.join(base_dir, noise_files.iloc[0])
    sr, sig_noise = wavfile.read(bg_noise_path)
    if sig_noise.ndim > 1: sig_noise = sig_noise[:, 0]

    # 2. بارگذاری صدای یک ماشین‌آلات مکانیکی (مثلاً قایق مسافربری یا موتور)
    target_files = df[df['Type'].str.lower() == 'passengers']['Filename'].dropna()
    target_path = os.path.join(base_dir, target_files.iloc[0])
    _, sig_target = wavfile.read(target_path)
    if sig_target.ndim > 1: sig_target = sig_target[:, 0]

    # برش 10 ثانیه اول برای نرمال‌سازی و سرعت بالا
    min_len = min(len(sig_noise), len(sig_target), int(10 * sr))
    sig_n = sig_noise[:min_len].astype(float)
    sig_t = sig_target[:min_len].astype(float)

    sig_n /= (np.max(np.abs(sig_n)) + 1e-10)
    sig_t /= (np.max(np.abs(sig_t)) + 1e-10)

    # 3. ترکیب سیگنال با نسبت 10 درصد (استتار شدید)
    test_ratio = 0.10
    sig_mix = (test_ratio * sig_t) + ((1.0 - test_ratio) * sig_n)

    print("Computing Bispectrum for Pure Ocean Noise...")
    B_noise, freqs = compute_bispectrum(sig_n, sr, nfft=2048, max_freq=1000)

    print("Computing Bispectrum for Pure Vessel...")
    B_target, _ = compute_bispectrum(sig_t, sr, nfft=2048, max_freq=1000)

    print(f"Computing Bispectrum for Mixed Signal (SNR: {test_ratio * 100}%)...")
    B_mix, _ = compute_bispectrum(sig_mix, sr, nfft=2048, max_freq=1000)

    # 4. ترسیم کانتورهای دوگانه فرکانسی
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style="white")

    # نرمال‌سازی مقادیر برای نمایش بهتر رنگ‌ها
    def plot_contour(ax, B_matrix, title):
        B_norm = B_matrix / (np.max(B_matrix) + 1e-15)
        X, Y = np.meshgrid(freqs, freqs)
        contour = ax.contourf(X, Y, B_norm, levels=30, cmap='magma')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency f1 (Hz)')
        ax.set_ylabel('Frequency f2 (Hz)')
        return contour

    plot_contour(axes[0], B_noise, 'Pure Ocean Noise (Gaussian)\nExpected: Near Zero Coupling')
    plot_contour(axes[1], B_target, 'Pure Vessel Signature\nExpected: High Phase Coupling')
    plot_contour(axes[2], B_mix, f'Mixed Signal ({int(test_ratio * 100)}% Vessel)\nAI/Fractal Blind Spot')

    plt.tight_layout()
    output_plot = os.path.join(output_dir, "Bispectrum_Contour_Analysis.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"\nDone! Bispectrum contours saved to: {output_plot}")


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
run_bispectrum_analysis(base_directory, code_directory)