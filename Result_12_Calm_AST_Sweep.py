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


def run_corrected_calm_ast_sweep(base_dir, code_dir):
    output_dir = os.path.join(code_dir, "Result_12_Calm_AST_Sweep_Corrected")
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)

    # 1. فیلتر کردن شرایط متلاطم و انتخاب یک محیط کاملاً آرام
    all_noise_files = df[df['Type'].str.lower() == 'natural ambient noise']['Filename'].dropna().tolist()
    extreme_conditions = ['lluvia', 'viento', 'oleaje', 'corriente']
    calm_noise_files = [f for f in all_noise_files if not any(ext in f.lower() for ext in extreme_conditions)]

    # استفاده از یک فایل مشخص به عنوان بستر عملیات
    bg_noise_filename = calm_noise_files[0]
    bg_noise_path = os.path.join(base_dir, bg_noise_filename)
    sr_bg, sig_bg = wavfile.read(bg_noise_path)
    if sig_bg.ndim > 1: sig_bg = sig_bg[:, 0]

    print(f"Operating Environment: {bg_noise_filename}")

    # 2. کالیبراسیون دینامیک: استخراج پروفایل صرفاً از همین محیط
    bg_hfds, bg_zcrs = [], []
    win_samples = int(0.5 * sr_bg)
    for i in range(0, len(sig_bg) - win_samples, win_samples):
        window = sig_bg[i:i + win_samples].astype(float)
        if np.max(np.abs(window)) == 0: continue
        bg_hfds.append(higuchi_fd(window, kmax=10))
        bg_zcrs.append(zero_crossing_rate(window))

    mu_hfd, std_hfd = np.mean(bg_hfds), np.std(bg_hfds)
    mu_zcr, std_zcr = np.mean(bg_zcrs), np.std(bg_zcrs)

    thresh_hfd = mu_hfd - (3 * std_hfd)
    thresh_zcr = mu_zcr - (3 * std_zcr)

    print(f"Corrected HFD Profile -> Mean: {mu_hfd:.4f} | Std: {std_hfd:.4f} | Threshold: {thresh_hfd:.4f}")
    print(f"Corrected ZCR Profile -> Mean: {mu_zcr:.4f} | Std: {std_zcr:.4f} | Threshold: {thresh_zcr:.4f}\n")

    ratios = np.round(np.arange(0.05, 0.55, 0.05), 2)
    target_classes = [c for c in df['Type'].dropna().unique() if c.lower() != 'natural ambient noise']

    summary_results = []

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

            if (mix_hfd < thresh_hfd) or (mix_zcr < thresh_zcr):
                detected_count += 1

        summary_results.append({'Target_Ratio': ratio, 'Detection_Count': detected_count})
        print(f"Ratio {ratio:.2f} ({(ratio) * 100:.0f}%) -> Detected: {detected_count} / 12")

    df_summary = pd.DataFrame(summary_results)
    df_summary.to_excel(os.path.join(output_dir, "Calm_AST_Sweep_Summary_Corrected.xlsx"), index=False)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df_summary, x='Target_Ratio', y='Detection_Count', marker='o', linewidth=3, color='teal')
    plt.axhline(y=12, color='green', linestyle='--', label='100% Detection')
    plt.title('Corrected Detection Rate vs. SNR (True Calm Baseline)')
    plt.xlabel('Target Signal Ratio (0.05 = 5%)')
    plt.ylabel('Number of Detected Vessels (Max 12)')
    plt.yticks(range(0, 13))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Calm_Detection_Rate_Sweep_Corrected.png"), dpi=300)
    plt.close()


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
run_corrected_calm_ast_sweep(base_directory, code_directory)