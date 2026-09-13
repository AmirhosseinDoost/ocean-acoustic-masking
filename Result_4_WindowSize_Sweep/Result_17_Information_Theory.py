import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.signal import butter, filtfilt, resample


def highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


def permutation_entropy(time_series, m=4, delay=1):
    """
    محاسبه آنتروپی جایگشتی.
    m: طول الگوی جایگشت (Embedding Dimension)
    delay: تاخیر زمانی (Time Delay)
    """
    n = len(time_series)
    permutations = list(itertools.permutations(range(m)))
    c = {perm: 0 for perm in permutations}

    for i in range(n - delay * (m - 1)):
        sorted_index_tuple = tuple(np.argsort(time_series[i:i + delay * m:delay]))
        c[sorted_index_tuple] += 1

    counts = [val for val in c.values() if val > 0]
    p = np.array(counts) / sum(counts)
    pe = -np.sum(p * np.log2(p))

    # نرمال‌سازی بین 0 (کاملاً منظم) و 1 (کاملاً آشوبناک)
    pe_normalized = pe / np.log2(math.factorial(m))
    return pe_normalized


import math


def run_entropy_analysis(base_dir_ships, deepwater_file, code_dir):
    output_dir = os.path.join(code_dir, "Result_17_Information_Theory")
    os.makedirs(output_dir, exist_ok=True)

    # بارگذاری نویز عمیق
    sig_noise, sr_noise = sf.read(deepwater_file)
    if sig_noise.ndim > 1: sig_noise = sig_noise[:, 0]

    # بارگذاری صدای شناور
    excel_path = os.path.join(base_dir_ships, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)
    target_files = df[df['Type'].str.lower() == 'passengers']['Filename'].dropna()
    target_path = os.path.join(base_dir_ships, target_files.iloc[0])
    sr_target, sig_target = wavfile.read(target_path)
    if sig_target.ndim > 1: sig_target = sig_target[:, 0]

    if sr_noise != sr_target:
        duration = min(len(sig_noise) / sr_noise, len(sig_target) / sr_target, 20.0)
        num_samples_target = int(duration * sr_noise)
        sig_target = resample(sig_target[:int(duration * sr_target)], num_samples_target)
        sig_noise = sig_noise[:num_samples_target]
        sr = sr_noise
    else:
        min_len = min(len(sig_noise), len(sig_target), int(20 * sr_noise))
        sig_noise = sig_noise[:min_len]
        sig_target = sig_target[:min_len]
        sr = sr_noise

    sig_n = highpass_filter(sig_noise.astype(float), cutoff=20.0, fs=sr)
    sig_t = highpass_filter(sig_target.astype(float), cutoff=20.0, fs=sr)

    sig_n /= (np.max(np.abs(sig_n)) + 1e-10)
    sig_t /= (np.max(np.abs(sig_t)) + 1e-10)

    # استخراج آنتروپی در پنجره‌های متوالی (Time-Series Entropy Profile)
    win_len = int(1.0 * sr)  # پنجره‌های ۱ ثانیه‌ای
    ratios = [0.0, 0.05, 0.10, 0.20, 0.50]  # درصدهای نفوذ از صفر مطلق تا 50 درصد

    results = []

    print("Extracting Entropy Profiles (Information Order)...")
    for ratio in ratios:
        sig_mix = (ratio * sig_t) + ((1.0 - ratio) * sig_n)

        ent_profile = []
        for i in range(0, len(sig_mix) - win_len, win_len):
            window = sig_mix[i:i + win_len]
            # برای سرعت و جلوگیری از کرش، روی زیرمجموعه‌ای از پنجره محاسبه می‌کنیم
            pe = permutation_entropy(window[::10], m=4, delay=1)
            ent_profile.append(pe)

        mean_ent = np.mean(ent_profile)
        std_ent = np.std(ent_profile)
        results.append({'SNR_Ratio': f"{int(ratio * 100)}%", 'Mean_Entropy': mean_ent, 'Std_Entropy': std_ent,
                        'Profile': ent_profile})
        print(f"SNR {int(ratio * 100):2d}% -> Mean Permutation Entropy: {mean_ent:.5f}")

    # ترسیم توزیع آنتروپی
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    colors = ['navy', 'purple', 'crimson', 'darkorange', 'gold']

    for idx, res in enumerate(results):
        sns.kdeplot(res['Profile'], color=colors[idx], label=f"Mixed {res['SNR_Ratio']} Vessel", linewidth=2.5,
                    fill=True, alpha=0.1)

    plt.title(
        'Information Theory: Permutation Entropy (PE) of Acoustic Signal\n(Lower Entropy = Higher Mechanical Order)')
    plt.xlabel('Normalized Permutation Entropy (1 = Pure Chaos/Randomness)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    output_plot = os.path.join(output_dir, "Entropy_Distribution_Sweep.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"\nDone! Entropy distributions saved to: {output_plot}")


base_ships = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
deep_file = r"D:\Academic Projects\Passive\Dataset\ADEON_Data\deepwater_calm.flac"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"

run_entropy_analysis(base_ships, deep_file, code_directory)