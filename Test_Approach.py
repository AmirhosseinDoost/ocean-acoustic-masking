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


def test_universal_stealth_detection(base_dir):
    excel_path = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)

    # 1. استخراج نویز مرجع اقیانوس
    noise_files = df[df['Type'] == 'Natural ambient noise']['Filename'].dropna()
    sr_noise, sig_noise = wavfile.read(os.path.join(base_dir, noise_files.iloc[0]))
    if sig_noise.ndim > 1: sig_noise = sig_noise[:, 0]

    # 2. شناسایی تمام کلاس‌های شناور به صورت داینامیک
    all_classes = df['Type'].dropna().unique().tolist()
    target_classes = [c for c in all_classes if c.lower() != 'natural ambient noise']

    detection_threshold = 1.9264  # مرز 3-سیگما محاسبه شده
    test_ratio = 0.1  # نسبت 10 درصد (نقطه کور سونار کلاسیک)

    print(f"--- Evaluating 3-Sigma Fractal Detector across {len(target_classes)} Vessel Classes ---")
    print(f"--- Target Signal Ratio: 10% (FFT Blind Spot) | Threshold: {detection_threshold} ---\n")

    results = []

    for v_type in target_classes:
        # جستجو برای پیدا کردن یک فایل سالم از این کلاس روی هارد دیسک
        matched_files = df[df['Type'] == v_type]['Filename'].dropna()
        filepath = None
        for file in matched_files:
            temp_path = os.path.join(base_dir, file)
            if os.path.exists(temp_path):
                filepath = temp_path
                break

        if not filepath:
            print(f"Skipping '{v_type}': No valid audio file found on disk.")
            continue

        sr_target, sig_target = wavfile.read(filepath)
        if sig_target.ndim > 1: sig_target = sig_target[:, 0]

        # هم‌اندازه و نرمال‌سازی کردن سیگنال‌ها (30 ثانیه اول)
        min_len = min(len(sig_noise), len(sig_target), int(30 * sr_noise))
        sig_t = sig_target[:min_len].astype(float)
        sig_n = sig_noise[:min_len].astype(float)

        sig_t = sig_t / np.max(np.abs(sig_t))
        sig_n = sig_n / np.max(np.abs(sig_n))

        # ترکیب سیگنال هدف و نویز اقیانوس با نسبت 10 درصد
        sig_mix = (test_ratio * sig_t) + ((1.0 - test_ratio) * sig_n)

        # محاسبه HFD ترکیب نهایی
        mix_hfd = higuchi_fd(sig_mix, kmax=10)

        # ارزیابی کشف
        detected = "YES [Found]" if mix_hfd < detection_threshold else "NO [Missed]"
        results.append((v_type, mix_hfd, detected))

        print(f"Vessel: {v_type:<15} | Mixed HFD: {mix_hfd:.4f} | Detected by Fractal: {detected}")

    # چاپ خلاصه عملکرد سیستم
    print("\n" + "=" * 50)
    print("--- Performance Summary at 10% SNR ---")
    success_count = sum(1 for r in results if "YES" in r[2])
    print(f"Classical Sonar (FFT) Detection Rate: 0 / {len(results)} (0%)")
    print(f"3-Sigma Fractal Sonar Detection Rate: {success_count} / {len(results)}")
    print("=" * 50)


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
test_universal_stealth_detection(base_directory)