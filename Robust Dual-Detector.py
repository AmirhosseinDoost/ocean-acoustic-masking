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


def zero_crossing_rate(x):
    # محاسبه ساده و سریع نرخ عبور از صفر با نامپای
    return ((x[:-1] * x[1:]) < 0).sum() / len(x)


def run_robust_dual_detector(base_dir, code_dir):
    output_dir = os.path.join(code_dir, "Result_10_Robust_Dual_Detector")
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)

    # 1. استخراج پروفایل جامع نویز از تمام 12 فایل
    noise_filenames = df[df['Type'].str.lower() == 'natural ambient noise']['Filename'].dropna().tolist()

    global_noise_hfds = []
    global_noise_zcrs = []

    print(f"--- Analyzing {len(noise_filenames)} Global Ocean Noise Files ---")
    for n_file in noise_filenames:
        n_path = os.path.join(base_dir, n_file)
        if not os.path.exists(n_path): continue

        sr_n, sig_n = wavfile.read(n_path)
        if sig_n.ndim > 1: sig_n = sig_n[:, 0]

        # تحلیل پنجره‌های 0.5 ثانیه‌ای برای استخراج رفتار آماری
        win_samples = int(0.5 * sr_n)
        for i in range(0, len(sig_n) - win_samples, win_samples):
            window = sig_n[i:i + win_samples].astype(float)
            if np.max(np.abs(window)) == 0: continue
            global_noise_hfds.append(higuchi_fd(window, kmax=10))
            global_noise_zcrs.append(zero_crossing_rate(window))

    # محاسبه خط‌کش‌های 3-سیگما برای HFD و ZCR
    mu_hfd, sig_hfd = np.mean(global_noise_hfds), np.std(global_noise_hfds)
    mu_zcr, sig_zcr = np.mean(global_noise_zcrs), np.std(global_noise_zcrs)

    thresh_hfd = mu_hfd - (3 * sig_hfd)
    thresh_zcr = mu_zcr - (3 * sig_zcr)

    print(f"\nRobust HFD Profile -> Mean: {mu_hfd:.4f}, Std: {sig_hfd:.4f} | 3-Sigma Threshold: {thresh_hfd:.4f}")
    print(f"Robust ZCR Profile -> Mean: {mu_zcr:.4f}, Std: {sig_zcr:.4f} | 3-Sigma Threshold: {thresh_zcr:.4f}\n")

    # 2. تست آشکارساز دوگانه روی 12 شناور در محیط ترکیبی
    target_classes = [c for c in df['Type'].dropna().unique() if c.lower() != 'natural ambient noise']
    test_ratio = 0.1  # نسبت 10% (نقطه کور FFT)

    # استفاده از یکی از نویزها به عنوان پس‌زمینه ترکیب (مثلاً نویز دارای جریان آب)
    bg_noise_path = os.path.join(base_dir, noise_filenames[0])
    sr_bg, sig_bg = wavfile.read(bg_noise_path)
    if sig_bg.ndim > 1: sig_bg = sig_bg[:, 0]

    results = []

    print(f"--- Testing Dual-Detector Logic (HFD < {thresh_hfd:.4f} OR ZCR < {thresh_zcr:.4f}) ---")
    for v_type in target_classes:
        matched_files = df[df['Type'] == v_type]['Filename'].dropna()
        filepath = next((os.path.join(base_dir, f) for f in matched_files if os.path.exists(os.path.join(base_dir, f))),
                        None)

        if not filepath: continue

        sr_t, sig_t = wavfile.read(filepath)
        if sig_t.ndim > 1: sig_t = sig_t[:, 0]

        min_len = min(len(sig_bg), len(sig_t), int(30 * sr_bg))
        s_t = sig_t[:min_len].astype(float)
        s_n = sig_bg[:min_len].astype(float)

        s_t /= np.max(np.abs(s_t)) + 1e-10
        s_n /= np.max(np.abs(s_n)) + 1e-10

        sig_mix = (test_ratio * s_t) + ((1.0 - test_ratio) * s_n)

        mix_hfd = higuchi_fd(sig_mix, kmax=10)
        mix_zcr = zero_crossing_rate(sig_mix)

        # منطق دوگانه: اگر حداقل یکی از ویژگی‌ها مرز 3-سیگما را بشکند، هدف کشف شده است
        det_hfd = mix_hfd < thresh_hfd
        det_zcr = mix_zcr < thresh_zcr
        detected = "YES" if (det_hfd or det_zcr) else "NO"

        results.append({
            'Vessel Type': v_type,
            'Mixed HFD': mix_hfd,
            'Mixed ZCR': mix_zcr,
            'HFD Trigger': det_hfd,
            'ZCR Trigger': det_zcr,
            'Final Detection': detected
        })
        print(f"{v_type:<15} | HFD: {mix_hfd:.4f} ({det_hfd}) | ZCR: {mix_zcr:.4f} ({det_zcr}) | Result: {detected}")

    # ذخیره نتایج
    df_results = pd.DataFrame(results)
    output_excel = os.path.join(output_dir, "Robust_Dual_Detector_Results.xlsx")
    df_results.to_excel(output_excel, index=False)

    success_count = sum(1 for r in results if r['Final Detection'] == "YES")
    print("\n" + "=" * 50)
    print("--- Robust Dual-Detector Summary at 10% SNR ---")
    print(f"Total Detection Rate: {success_count} / {len(results)}")
    print(f"Results saved to: {output_excel}")
    print("=" * 50)


# مسیرها را مطابق سیستم خود تنظیم کنید
base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"

run_robust_dual_detector(base_directory, code_directory)