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


def get_env_label(filename):
    fname_lower = filename.lower()
    if 'lluvia' in fname_lower: return 'Rain'
    if 'viento' in fname_lower: return 'Wind'
    if 'oleaje' in fname_lower: return 'Waves'
    if 'corriente' in fname_lower: return 'Current'
    return f"Calm ({filename.split('__')[0]})"


def run_dynamic_ast_matrix(base_dir, code_dir):
    output_dir = os.path.join(code_dir, "Result_13_Dynamic_Environment_AST")
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(base_dir, "shipsEar.xlsx")
    df = pd.read_excel(excel_path)

    noise_files = df[df['Type'].str.lower() == 'natural ambient noise']['Filename'].dropna().tolist()
    target_classes = [c for c in df['Type'].dropna().unique() if c.lower() != 'natural ambient noise']
    ratios = np.round(np.arange(0.05, 0.55, 0.05), 2)

    # تنظیم ماتریس خروجی
    env_labels = [get_env_label(f) for f in noise_files]
    ast_matrix = pd.DataFrame(index=target_classes, columns=env_labels)

    print("Starting Dynamic CFAR Analysis (144 Combinations)...")

    for n_idx, n_file in enumerate(noise_files):
        env_name = env_labels[n_idx]
        n_path = os.path.join(base_dir, n_file)
        if not os.path.exists(n_path): continue

        sr_bg, sig_bg = wavfile.read(n_path)
        if sig_bg.ndim > 1: sig_bg = sig_bg[:, 0]

        # 1. کالیبراسیون دینامیک برای همین محیط
        bg_hfds, bg_zcrs = [], []
        win_samples = int(0.5 * sr_bg)
        for i in range(0, len(sig_bg) - win_samples, win_samples):
            window = sig_bg[i:i + win_samples].astype(float)
            if np.max(np.abs(window)) == 0: continue
            bg_hfds.append(higuchi_fd(window, kmax=10))
            bg_zcrs.append(zero_crossing_rate(window))

        t_hfd = np.mean(bg_hfds) - (3 * np.std(bg_hfds))
        t_zcr = np.mean(bg_zcrs) - (3 * np.std(bg_zcrs))

        print(f"\n[{env_name}] Thresholds -> HFD: {t_hfd:.4f} | ZCR: {t_zcr:.4f}")

        # 2. تست تمام شناورها در این محیط
        for v_type in target_classes:
            matched_files = df[df['Type'] == v_type]['Filename'].dropna()
            filepath = next(
                (os.path.join(base_dir, f) for f in matched_files if os.path.exists(os.path.join(base_dir, f))), None)
            if not filepath: continue

            sr_t, sig_t = wavfile.read(filepath)
            if sig_t.ndim > 1: sig_t = sig_t[:, 0]

            # 10 ثانیه اول برای افزایش سرعت محاسبات
            min_len = min(len(sig_bg), len(sig_t), int(10 * sr_bg))
            s_t = sig_t[:min_len].astype(float)
            s_n = sig_bg[:min_len].astype(float)

            s_t /= np.max(np.abs(s_t)) + 1e-10
            s_n /= np.max(np.abs(s_n)) + 1e-10

            # 3. پیدا کردن حداقل درصد کشف (AST)
            detection_point = np.nan  # اگر تا 50% کشف نشود، NaN می‌ماند
            for ratio in ratios:
                sig_mix = (ratio * s_t) + ((1.0 - ratio) * s_n)
                mix_hfd = higuchi_fd(sig_mix, kmax=10)
                mix_zcr = zero_crossing_rate(sig_mix)

                if (mix_hfd < t_hfd) or (mix_zcr < t_zcr):
                    detection_point = ratio
                    break  # هدف پیدا شد، به درصد بالاتر نیاز نیست

            ast_matrix.at[v_type, env_name] = detection_point
            status = f"{int(detection_point * 100)}%" if not pd.isna(detection_point) else ">50%"
            print(f"  -> {v_type:<15} : Discovered at {status} SNR")

    # ذخیره و رسم نقشه حرارتی
    ast_matrix = ast_matrix.apply(pd.to_numeric)
    ast_matrix.to_excel(os.path.join(output_dir, "AST_Dynamic_Matrix.xlsx"))

    plt.figure(figsize=(14, 8))
    # مقادیر NaN (کشف نشده تا 50%) را برای رسم با 0.55 پر می‌کنیم
    plot_matrix = ast_matrix.fillna(0.55) * 100

    sns.heatmap(plot_matrix, annot=True, fmt=".0f", cmap="YlOrRd_r", cbar_kws={'label': 'Detection Threshold (%)'})
    plt.title(
        'Acoustic Stealth Threshold (AST) Matrix Across Environments\n(Lower % = Better Sonar Detection | 55 = Undetected even at 50%)')
    plt.ylabel('Vessel Class')
    plt.xlabel('Ocean Environment')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AST_Heatmap.png"), dpi=300)
    plt.close()

    print(f"\nDone! Matrix and Heatmap saved to: {output_dir}")


base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"
run_dynamic_ast_matrix(base_directory, code_directory)