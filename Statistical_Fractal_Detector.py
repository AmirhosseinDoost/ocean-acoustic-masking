import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns


def higuchi_fd(x, kmax=10):
    N = len(x)
    L = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lk = np.zeros(k)
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) <= 1: continue
            Lmk = np.sum(np.abs(np.diff(x[idxs])))
            norm = (N - 1) / (idxs[-1] - m) / k
            Lk[m] = Lmk * norm
        L[k - 1] = np.mean(Lk)
    ln_L = np.log(L)
    ln_k = np.log(1 / np.arange(1, kmax + 1))
    slope, _ = np.polyfit(ln_k, ln_L, 1)
    return slope


def run_statistical_fractal_detector(code_dir):
    audio_dir = os.path.join(code_dir, "Result_5_Signal_Mixing", "Result_5_Mixed_Audio_Files")
    output_dir = os.path.join(code_dir, "Result_9_Statistical_Fractal_Detector")
    os.makedirs(output_dir, exist_ok=True)

    # 1. استخراج پروفایل آماری نویز خالص اقیانوس (0% Target)
    noise_file = os.path.join(audio_dir, "Mixed_0pct_Target_100pct_Noise.wav")
    sr, sig_noise = wavfile.read(noise_file)

    # تقسیم نویز به پنجره‌های 0.5 ثانیه‌ای برای محاسبه انحراف معیار طبیعی
    window_samples = int(0.5 * sr)
    noise_hfds = []
    for i in range(0, len(sig_noise) - window_samples, window_samples):
        window = sig_noise[i:i + window_samples]
        noise_hfds.append(higuchi_fd(window, kmax=10))

    mu_noise = np.mean(noise_hfds)
    sigma_noise = np.std(noise_hfds)
    detection_threshold = mu_noise - (3 * sigma_noise)

    print(f"Ocean Noise Profile -> Mean (mu): {mu_noise:.4f}, Std (sigma): {sigma_noise:.4f}")
    print(f"Fractal Detection Threshold (mu - 3*sigma): {detection_threshold:.4f}")

    # 2. تست آشکارساز روی سیگنال‌های ترکیبی
    ratios = np.round(np.arange(0.0, 1.1, 0.1), 1)
    results = []

    for alpha in ratios:
        t_pct = int(alpha * 100)
        n_pct = int((1.0 - alpha) * 100)
        filepath = os.path.join(audio_dir, f"Mixed_{t_pct}pct_Target_{n_pct}pct_Noise.wav")

        _, sig_mix = wavfile.read(filepath)
        mix_hfd = higuchi_fd(sig_mix, kmax=10)  # HFD کل سیگنال

        detected = "Yes" if mix_hfd < detection_threshold else "No"
        results.append({
            'Target_Ratio': alpha,
            'Mixed_HFD': mix_hfd,
            'Fractal_Detected': detected
        })

    df_results = pd.DataFrame(results)
    df_results.to_excel(os.path.join(output_dir, "Fractal_Detector_Results.xlsx"), index=False)

    # 3. رسم نمودار مقایسه‌ای
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.lineplot(data=df_results, x='Target_Ratio', y='Mixed_HFD', marker='o', color='darkred', linewidth=3,
                 label='Mixed Signal HFD')

    # رسم محدوده نویز طبیعی و خط آستانه کشف
    plt.axhline(y=mu_noise, color='green', linestyle='--', linewidth=2, label=f'Noise Mean ($\mu$) = {mu_noise:.2f}')
    plt.axhspan(mu_noise - (3 * sigma_noise), mu_noise + (3 * sigma_noise), color='green', alpha=0.1,
                label='Normal Ocean Fluctuation (3$\sigma$)')
    plt.axhline(y=detection_threshold, color='red', linestyle='-.', linewidth=2,
                label=f'Detection Threshold ($\mu - 3\sigma$) = {detection_threshold:.2f}')

    # مشخص کردن نقطه کور سونار کلاسیک (از آزمایش قبلی روی نسبت 0.1)
    plt.axvline(x=0.1, color='purple', linestyle=':', linewidth=2, label='Classical FFT Blind Spot (0.1)')

    plt.gca().invert_xaxis()
    plt.title('Statistical Fractal Sonar: Absolute Detection Threshold')
    plt.xlabel('Target Signal Ratio (1.0 = Pure Target, 0.0 = Pure Noise)')
    plt.ylabel('Higuchi Fractal Dimension (HFD)')
    plt.legend(loc='lower left')
    plt.tight_layout()

    output_plot = os.path.join(output_dir, "Fractal_3Sigma_Detection.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"\nSuccess! Detector profile saved to: {output_plot}")


code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
run_statistical_fractal_detector(code_directory)

#%% Block 2
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile


def higuchi_fd(x, kmax=10):
    N = len(x)
    L = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lk = np.zeros(k)
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) <= 1: continue
            Lmk = np.sum(np.abs(np.diff(x[idxs])))
            norm = (N - 1) / (idxs[-1] - m) / k
            Lk[m] = Lmk * norm
        L[k - 1] = np.mean(Lk)
    ln_L = np.log(L)
    ln_k = np.log(1 / np.arange(1, kmax + 1))
    slope, _ = np.polyfit(ln_k, ln_L, 1)
    return slope


def test_multiclass_stealth_detection(base_dir):
    excel_path = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)

    # فایل نویز مرجع
    noise_filename = df[df['Type'] == 'Natural ambient noise']['Filename'].dropna().iloc[0]
    sr_noise, sig_noise = wavfile.read(os.path.join(base_dir, noise_filename))
    if sig_noise.ndim > 1: sig_noise = sig_noise[:, 0]

    # کلاس‌های مختلف برای تست
    target_classes = ['Cargo', 'Passengers', 'Motorboat']

    detection_threshold = 1.9264  # مرز محاسبه شده در مرحله قبل
    test_ratio = 0.1  # نسبت 10 درصد (نقطه کور سونار کلاسیک)

    print(f"--- Testing Statistical Fractal Detector (Threshold: {detection_threshold}) ---")
    print(f"--- Target Signal Ratio: {test_ratio * 100}% (Classical FFT Blind Spot) ---\n")

    results = []

    for v_type in target_classes:
        filename = df[df['Type'] == v_type]['Filename'].dropna().iloc[0]
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath):
            print(f"File for {v_type} not found. Skipping...")
            continue

        sr_target, sig_target = wavfile.read(filepath)
        if sig_target.ndim > 1: sig_target = sig_target[:, 0]

        # هم‌اندازه کردن و نرمال‌سازی سیگنال‌ها (30 ثانیه اول)
        min_len = min(len(sig_noise), len(sig_target), int(30 * sr_noise))
        sig_t = sig_target[:min_len].astype(float)
        sig_n = sig_noise[:min_len].astype(float)

        sig_t = sig_t / np.max(np.abs(sig_t))
        sig_n = sig_n / np.max(np.abs(sig_n))

        # ترکیب سیگنال با نسبت 10 درصد
        sig_mix = (test_ratio * sig_t) + ((1.0 - test_ratio) * sig_n)

        # محاسبه HFD ترکیب نهایی
        mix_hfd = higuchi_fd(sig_mix, kmax=10)

        # ارزیابی کشف
        detected = "YES [Target Found!]" if mix_hfd < detection_threshold else "NO [Missed]"

        results.append({
            'Vessel Type': v_type,
            'Mixed HFD': round(mix_hfd, 4),
            'Detected': detected
        })

        print(f"Vessel: {v_type:<12} | Mixed HFD: {mix_hfd:.4f} | Detected by Fractal: {detected}")


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
test_multiclass_stealth_detection(base_directory)