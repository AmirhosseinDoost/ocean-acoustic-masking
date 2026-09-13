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


def run_window_size_sweep(base_dir, code_dir):
    # ایجاد پوشه مجزا برای نتایج تحلیل پارامتریک طول پنجره
    output_dir = os.path.join(code_dir, "Result_4_WindowSize_Sweep")
    os.makedirs(output_dir, exist_ok=True)

    original_excel = os.path.join(base_dir, "shipsEar.xlsx")
    output_excel = os.path.join(output_dir, "WindowSize_Sweep_Results.xlsx")

    df = pd.read_excel(original_excel)

    # مقادیر مختلف طول پنجره (بر حسب میلی‌ثانیه)
    window_sizes_ms = [10, 25, 50, 100, 250, 500, 1000]
    target_classes = ['Dredger', 'Passengers', 'RORO', 'Natural ambient noise']
    selected_files = {}

    # انتخاب یک فایل نماینده از هر کلاس
    for index, row in df.iterrows():
        vessel_type = row.get('Type')
        filename = row.get('Filename')
        if vessel_type in target_classes and vessel_type not in selected_files:
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath): selected_files[vessel_type] = filepath

    results = []

    for v_type, filepath in selected_files.items():
        print(f"Sweeping window sizes for: {v_type}")
        sample_rate, data = wavfile.read(filepath)
        if data.ndim > 1: data = data[:, 0]

        # استفاده از 30 ثانیه اول برای داشتن داده کافی در پنجره‌های بزرگ (1000ms)
        data = data[:min(len(data), 30 * sample_rate)]

        for w_ms in window_sizes_ms:
            window_size_sec = w_ms / 1000.0
            window_samples = int(window_size_sec * sample_rate)

            if window_samples == 0 or window_samples > len(data):
                continue

            num_windows = len(data) // window_samples
            hfd_list = []

            for i in range(num_windows):
                window = data[i * window_samples: (i + 1) * window_samples]
                if np.max(np.abs(window)) > 0:
                    hfd_list.append(higuchi_fd(window, kmax=10))  # kmax ثابت نگه داشته می‌شود

            mean_hfd = np.mean(hfd_list) if hfd_list else np.nan
            results.append({'Type': v_type, 'Window_Size_ms': w_ms, 'Mean_HFD': mean_hfd})

    # ذخیره داده‌ها
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_excel, index=False)

    # رسم نمودار
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # استفاده از محور X به صورت لگاریتمی برای نمایش بهتر فاصله‌های زمانی
    ax = sns.lineplot(data=df_results, x='Window_Size_ms', y='Mean_HFD', hue='Type', marker='o', linewidth=2)
    ax.set_xscale('log')
    ax.set_xticks(window_sizes_ms)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.title('Parametric Sweep: Effect of Time Window Size on HFD')
    plt.xlabel('Window Size (Milliseconds) - Log Scale')
    plt.ylabel('Mean Higuchi Fractal Dimension')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "WindowSize_Parametric_Sweep.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\nSuccess! Results and plot saved to: {output_dir}")


# مسیرهای اجرای کد
base_directory = r"D:\Academic Projects\Passive\Dataset\drive-download-20260901T040217Z-1-001"
code_directory = r"D:\Academic Projects\Passive\Dataset\Code"

run_window_size_sweep(base_directory, code_directory)