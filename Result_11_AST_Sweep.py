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


def zero_crossing_rate(x):
    return ((x[:-1] * x[1:]) < 0).sum() / len(x)


def run_ast_sweep(base_dir, code_dir):
    output_dir = os.path.join(code_dir, "Result_11_AST_Sweep")
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)

    # استخراج پروفایل مرجع از تمام 12 فایل نویز
    noise_filenames = df[df['Type'].str.lower() == 'natural ambient noise']['Filename'].dropna().tolist()
    global_noise_hfds, global_noise_zcrs = [], []

    print("Calculating global noise profile...")
    for n_file in noise_filenames:
        n_path = os.path.join(base_dir, n_file)
        if not os.path.exists(n_path): continue
        sr_n, sig_n = wavfile.read(n_path)
        if sig_n.ndim > 1: sig_n = sig_n[:, 0]

        win_samples = int(0.5 * sr_n)
        for i in range(0, len(sig_n) - win_samples, win_samples):
            window = sig_n[i:i + win_samples].astype(float)
            if np.max(np.abs(window)) == 0: continue
            global_noise_hfds.append(higuchi_fd(window, kmax=10))
            global_noise_zcrs.append(zero_crossing_rate(window))

    thresh_hfd = np.mean(global_noise_hfds) - (3 * np.std(global_noise_hfds))
    thresh_zcr = np.mean(global_noise_zcrs) - (3 * np.std(global_noise_zcrs))

    # ایجاد بازه درصدهای مختلف برای تست (از 10% تا 50%)
    ratios = np.round(np.arange(0.10, 0.55, 0.05), 2)
    target_classes = [c for c in df['Type'].dropna().unique() if c.lower() != 'natural ambient noise']

    bg_noise_path = os.path.join(base_dir, noise_filenames[0])
    sr_bg, sig_bg = wavfile.read(bg_noise_path)
    if sig_bg.ndim > 1: sig_bg = sig_bg[:, 0]

    summary_results = []
    detailed_results = []

    print(f"Thresholds -> HFD: {thresh_hfd:.4f} | ZCR: {thresh_zcr:.4f}")
    print("Starting SNR Sweep Analysis...\n")

    for ratio in ratios:
        detected_count = 0
        for v_type in target_classes:
            matched_files = df[df['Type'] == v_type]['Filename'].dropna()
            filepath = next(
                (os.path.join(base_dir, f) for f in matched_files if os.path.exists(os.path.join(base_dir, f))), None)
            if not filepath: continue

            sr_t, sig_t = wavfile.read(filepath)
            if sig_t.ndim > 1: sig_t = sig_t[:, 0]

            min_len = min(len(sig_bg), len(sig_t), int(30 * sr_bg))
            s_t = sig_t[:min_len].astype(float)
            s_n = sig_bg[:min_len].astype(float)

            s_t /= np.max(np.abs(s_t)) + 1e-10
            s_n /= np.max(np.abs(s_n)) + 1e-10

            sig_mix = (ratio * s_t) + ((1.0 - ratio) * s_n)

            mix_hfd = higuchi_fd(sig_mix, kmax=10)
            mix_zcr = zero_crossing_rate(sig_mix)

            # منطق کشف دوگانه
            detected = (mix_hfd < thresh_hfd) or (mix_zcr < thresh_zcr)
            if detected: detected_count += 1

            detailed_results.append({'Ratio': ratio, 'Vessel': v_type, 'Detected': detected})

        summary_results.append({'Target_Ratio': ratio, 'Detection_Count': detected_count})
        print(f"Ratio {ratio:.2f} ({(ratio) * 100:.0f}%) -> Detected: {detected_count} / 12")

    # ذخیره داده‌ها و رسم نمودار
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_excel(os.path.join(output_dir, "AST_Sweep_Summary.xlsx"), index=False)
    pd.DataFrame(detailed_results).to_excel(os.path.join(output_dir, "AST_Sweep_Detailed.xlsx"), index=False)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df_summary, x='Target_Ratio', y='Detection_Count', marker='o', linewidth=3, color='navy')
    plt.axhline(y=12, color='green', linestyle='--', label='100% Detection (12/12)')

    plt.title('Detection Rate vs. Target Signal Ratio (Robust Dual-Detector)')
    plt.xlabel('Target Signal Ratio (0.10 = 10%)')
    plt.ylabel('Number of Detected Vessels (Max 12)')
    plt.yticks(range(0, 13))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Detection_Rate_Sweep.png"), dpi=300)
    plt.close()

    print(f"\nDone! Results and plot saved to: {output_dir}")


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
run_ast_sweep(base_directory, code_directory)